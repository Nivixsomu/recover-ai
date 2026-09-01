"""Pre-ML audit for the controlled synthetic RecoverAI recovery dataset.

This script performs descriptive and leakage checks only. It does not train a
predictive model and does not modify simulator outputs.
"""

from __future__ import annotations

from math import log2
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.data.schemas import SimulationConfig
from ml.src.data.simulator import _choose_action
from ml.src.data.validation import PREDICTION_FEATURE_COLUMNS, prediction_feature_matrix, validate_simulation


PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
FINAL_DIR = PROJECT_ROOT / "ml" / "data" / "final"
REPORT_PATH = PROJECT_ROOT / "docs" / "pre_ml_audit.md"


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a Markdown table without optional dependencies."""
    header = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def amount_band(amount: float) -> str:
    if amount <= 1_000:
        return "INR_100_1000"
    if amount <= 5_000:
        return "INR_1001_5000"
    if amount <= 25_000:
        return "INR_5001_25000"
    return "INR_25001_PLUS"


def observed_group_table(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    table = frame.groupby(group_columns, as_index=False).agg(
        cases=("recovery_case_id", "size"),
        recovery_rate=("recovery_success", "mean"),
        recovered_revenue=("recovered_amount", "sum"),
        average_amount_at_risk=("amount_at_risk", "mean"),
        average_recovered_amount=("recovered_amount", "mean"),
    )
    table["recovery_rate"] = (table["recovery_rate"] * 100).round(2)
    for column in ("recovered_revenue", "average_amount_at_risk", "average_recovered_amount"):
        table[column] = table[column].round(2)
    return table.sort_values(group_columns).reset_index(drop=True)


def binary_entropy(probability: float) -> float:
    if probability in (0.0, 1.0):
        return 0.0
    return -(probability * log2(probability) + (1 - probability) * log2(1 - probability))


def failure_reason_difficulty(observed: pd.DataFrame, potential: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Measure descriptive outcome association and best-action determinism."""
    outcome = observed.groupby("failure_reason", as_index=False).agg(cases=("recovery_case_id", "size"), recovery_rate=("recovery_success", "mean"))
    outcome["recovery_rate"] = (outcome["recovery_rate"] * 100).round(2)
    overall_success = float(observed["recovery_success"].mean())
    majority_accuracy = max(overall_success, 1 - overall_success)
    rates = observed.groupby("failure_reason")["recovery_success"].mean()
    reason_rule = observed["failure_reason"].map(rates).ge(0.5).astype(int)
    reason_rule_accuracy = float((reason_rule == observed["recovery_success"]).mean())
    overall_entropy = binary_entropy(overall_success)
    conditional_entropy = sum(
        len(group) / len(observed) * binary_entropy(float(group["recovery_success"].mean()))
        for _, group in observed.groupby("failure_reason")
    )
    normalized_entropy_reduction = (overall_entropy - conditional_entropy) / overall_entropy if overall_entropy else 0.0

    probability_pivot = potential.pivot(index="recovery_case_id", columns="potential_action", values="potential_recovery_probability")
    best = probability_pivot.idxmax(axis=1).rename("best_action_from_probability")
    best_frame = observed[["recovery_case_id", "failure_reason", "amount_at_risk", "retry_count"]].merge(best, left_on="recovery_case_id", right_index=True)
    best_by_reason = best_frame.groupby("failure_reason", as_index=False).agg(
        cases=("recovery_case_id", "size"),
        distinct_best_actions=("best_action_from_probability", "nunique"),
        modal_best_action=("best_action_from_probability", lambda values: values.mode().iat[0]),
        modal_share=("best_action_from_probability", lambda values: values.value_counts(normalize=True).iat[0]),
    )
    best_by_reason["modal_share"] = (best_by_reason["modal_share"] * 100).round(2)
    exact_reason_lookup_accuracy = float(
        best_frame.groupby("failure_reason")["best_action_from_probability"].transform(lambda values: values.mode().iat[0]).eq(best_frame["best_action_from_probability"]).mean()
    )
    diagnostic = {
        "majority_accuracy": majority_accuracy * 100,
        "reason_rule_accuracy": reason_rule_accuracy * 100,
        "entropy_reduction": normalized_entropy_reduction * 100,
        "best_action_lookup_accuracy": exact_reason_lookup_accuracy * 100,
    }
    return outcome.sort_values("failure_reason"), best_by_reason.sort_values("failure_reason"), best_frame, diagnostic


def action_effect_tables(observed: pd.DataFrame, potential: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mean_action = potential.groupby("potential_action", as_index=False).agg(
        mean_simulated_success_probability=("potential_recovery_probability", "mean"),
        mean_simulated_recovered_amount=("potential_recovered_amount", "mean"),
        simulated_success_rate=("potential_recovery_success", "mean"),
    )
    for column in ("mean_simulated_success_probability", "simulated_success_rate"):
        mean_action[column] = (mean_action[column] * 100).round(2)
    mean_action["mean_simulated_recovered_amount"] = mean_action["mean_simulated_recovered_amount"].round(2)

    comparable = observed[["recovery_case_id", "failure_reason", "amount_at_risk", "retry_count"]].copy()
    comparable["amount_band"] = comparable["amount_at_risk"].map(amount_band)
    comparable = comparable.merge(potential[["recovery_case_id", "potential_action", "potential_recovery_probability"]], on="recovery_case_id")
    comparable_table = comparable.groupby(["failure_reason", "amount_band", "retry_count", "potential_action"], as_index=False).agg(
        cases=("recovery_case_id", "size"),
        mean_probability=("potential_recovery_probability", "mean"),
    )
    comparable_table["mean_probability"] = (comparable_table["mean_probability"] * 100).round(2)
    by_failure_action = potential.merge(observed[["recovery_case_id", "failure_reason"]], on="recovery_case_id").groupby(
        ["failure_reason", "potential_action"], as_index=False
    ).agg(cases=("recovery_case_id", "size"), mean_probability=("potential_recovery_probability", "mean"))
    by_failure_action["mean_probability"] = (by_failure_action["mean_probability"] * 100).round(2)
    return mean_action.sort_values("potential_action"), by_failure_action.sort_values(["failure_reason", "potential_action"]), comparable_table.sort_values(["failure_reason", "amount_band", "retry_count", "potential_action"])


def baseline_test_audit(observed: pd.DataFrame) -> tuple[dict[str, float | int], pd.DataFrame, pd.DataFrame]:
    config = SimulationConfig()
    test = observed.loc[observed["split"] == "test"].copy()
    selected = test.apply(lambda row: _choose_action(row.to_dict(), config), axis=1)
    violations = int((selected != test["observed_action"]).sum())
    interventions = test["observed_action"] != "NO_ACTION"
    unrecovered_proxy = float((test.loc[interventions, "recovery_success"] == 0).mean())
    summary: dict[str, float | int] = {
        "cases": len(test),
        "recovery_rate": float(test["recovery_success"].mean() * 100),
        "total_recovered_revenue": float(test["recovered_amount"].sum()),
        "average_recovered_amount_all_cases": float(test["recovered_amount"].mean()),
        "average_recovered_amount_successes": float(test.loc[test["recovery_success"] == 1, "recovered_amount"].mean()),
        "unrecovered_intervention_rate_proxy": unrecovered_proxy * 100,
        "policy_violations": violations,
    }
    return summary, observed_group_table(test, ["failure_reason", "observed_action"]), observed_group_table(test, ["observed_action"])


def main() -> None:
    observed = pd.read_csv(PROCESSED_DIR / "recovery_observed.csv", parse_dates=["created_at", "customer_created_at", "last_history_event_at"])
    potential = pd.read_csv(PROCESSED_DIR / "recovery_potential_outcomes.csv")
    customers = pd.read_csv(PROCESSED_DIR / "recovery_customers.csv", parse_dates=["customer_created_at"])
    config = SimulationConfig()
    validate_simulation(observed, potential, customers, config)
    features = prediction_feature_matrix(observed)
    observed["amount_band"] = observed["amount_at_risk"].map(amount_band)

    distribution_tables = {
        "Payment methods": observed["payment_method"].value_counts().rename_axis("payment_method").reset_index(name="cases"),
        "Retry count": observed["retry_count"].value_counts().sort_index().rename_axis("retry_count").reset_index(name="cases"),
        "Customer segments": observed["customer_segment"].value_counts().rename_axis("customer_segment").reset_index(name="cases"),
        "Recovery success": observed["recovery_success"].value_counts().sort_index().rename_axis("recovery_success").reset_index(name="cases"),
    }
    human_review_pct = (observed["observed_action"] == "HUMAN_REVIEW").mean() * 100
    failure_action = observed_group_table(observed, ["failure_reason", "observed_action"])
    method_action = observed_group_table(observed, ["payment_method", "observed_action"])
    amount_action = observed_group_table(observed, ["amount_band", "observed_action"])
    retry_action = observed_group_table(observed, ["retry_count", "observed_action"])
    segment_action = observed_group_table(observed, ["customer_segment", "observed_action"])
    outcome_by_reason, best_by_reason, best_frame, difficulty = failure_reason_difficulty(observed, potential)
    mean_potential, potential_by_reason, comparable_potential = action_effect_tables(observed, potential)

    # Expected value uses the probability for the action actually selected by the baseline.
    selected_potential = potential.merge(
        observed[["recovery_case_id", "observed_action", "amount_at_risk"]],
        left_on=["recovery_case_id", "potential_action"],
        right_on=["recovery_case_id", "observed_action"],
        how="inner",
    )
    expected_revenue = float((selected_potential["potential_recovery_probability"] * selected_potential["amount_at_risk"]).sum())
    potential_values = potential.merge(observed[["recovery_case_id", "amount_at_risk"]], on="recovery_case_id")
    potential_values["expected_recovered_value"] = potential_values["potential_recovery_probability"] * potential_values["amount_at_risk"]
    best_value_rows = potential_values.loc[potential_values.groupby("recovery_case_id")["expected_recovered_value"].idxmax()].copy()
    optimal_expected_revenue = float(best_value_rows["expected_recovered_value"].sum())
    optimal_simulated_recovered_value = float(best_value_rows["potential_recovered_amount"].sum())
    regret_absolute = optimal_expected_revenue - expected_revenue
    regret_relative = regret_absolute / optimal_expected_revenue if optimal_expected_revenue else 0.0
    optimal_distribution = best_frame["best_action_from_probability"].value_counts().rename_axis("probability_optimal_action").reset_index(name="cases")
    optimal_distribution["percentage"] = (optimal_distribution["cases"] / len(best_frame) * 100).round(2)
    context_summary = best_frame.merge(
        observed[["recovery_case_id", "historical_success_rate", "payment_method", "customer_segment"]],
        on="recovery_case_id",
    ).groupby(["failure_reason", "best_action_from_probability"], as_index=False).agg(
        cases=("recovery_case_id", "size"),
        average_amount_at_risk=("amount_at_risk", "mean"),
        average_retry_count=("retry_count", "mean"),
        average_historical_success_rate=("historical_success_rate", "mean"),
    )
    context_summary["average_amount_at_risk"] = context_summary["average_amount_at_risk"].round(2)
    context_summary["average_retry_count"] = context_summary["average_retry_count"].round(2)
    context_summary["average_historical_success_rate"] = (context_summary["average_historical_success_rate"] * 100).round(2)
    multiple_action_reason_count = int((best_by_reason["distinct_best_actions"] > 1).sum())
    needs_revision = difficulty["best_action_lookup_accuracy"] >= 95 or multiple_action_reason_count < 3
    decision = "B. SIMULATOR NEEDS REVISION BEFORE ML" if needs_revision else "A. READY FOR ML"
    decision_explanation = (
        "The optimal action remains nearly deterministic from failure reason."
        if needs_revision
        else "Failure reason no longer determines the optimal action: multiple pre-intervention contexts change the probability-optimal action."
    )
    test_summary, test_failure_action, test_action = baseline_test_audit(observed)
    test_summary["expected_recovered_revenue"] = float(
        (selected_potential.loc[selected_potential["recovery_case_id"].isin(set(observed.loc[observed["split"] == "test", "recovery_case_id"])), "potential_recovery_probability"]
        * selected_potential.loc[selected_potential["recovery_case_id"].isin(set(observed.loc[observed["split"] == "test", "recovery_case_id"])), "amount_at_risk"]
        ).sum()
    )
    split_table = observed.groupby("split", as_index=False).agg(cases=("recovery_case_id", "size"), customers=("customer_id", "nunique"), earliest=("created_at", "min"), latest=("created_at", "max"))
    split_customer_sets = [set(observed.loc[observed["split"] == split, "customer_id"]) for split in ("train", "validation", "test")]
    no_customer_overlap = not any(split_customer_sets[left].intersection(split_customer_sets[right]) for left in range(3) for right in range(left + 1, 3))

    report = f"""# RecoverAI Pre-ML Audit

## Decision

**{decision}.** {decision_explanation} The result is assessed from the observed action distribution, within-failure optimal-action variation, and leakage checks; it is not evidence of real-world payment behavior.

This is a simulator finding only; no claim is made about real-world causal effects, Razorpay, banks, merchants, or customers.

## Dataset and economic overview

- Opportunities/payments: {len(observed):,}; customers: {observed['customer_id'].nunique():,}; potential outcomes: {len(potential):,}
- Seed/configuration: `{config.seed}` / `{config.version}`
- Total amount at risk: INR {observed['amount_at_risk'].sum():,.2f}
- Total recovered revenue: INR {observed['recovered_amount'].sum():,.2f}
- Observed recovery rate: {observed['recovery_success'].mean() * 100:.2f}%
- Average recovered amount, all opportunities: INR {observed['recovered_amount'].mean():,.2f}
- Average recovered amount, successful opportunities only: INR {observed.loc[observed['recovery_success'] == 1, 'recovered_amount'].mean():,.2f}
- Expected recovered revenue under selected simulated actions: INR {expected_revenue:,.2f}
- Intervention costs are not simulated; therefore this audit reports no profit or net-revenue claim.

## Distribution audit

Amount at risk is long-tailed rather than uniform: min INR {observed['amount_at_risk'].min():,.2f}, median INR {observed['amount_at_risk'].median():,.2f}, mean INR {observed['amount_at_risk'].mean():,.2f}, 75th percentile INR {observed['amount_at_risk'].quantile(.75):,.2f}, and max INR {observed['amount_at_risk'].max():,.2f}. Recovery is not extremely imbalanced ({observed['recovery_success'].mean() * 100:.2f}% success / {(1 - observed['recovery_success'].mean()) * 100:.2f}% failure). No impossible amounts, identifiers, categories, timestamps, or outcome relationships were found by validation.

### Payment methods

{markdown_table(distribution_tables['Payment methods'])}

### Retry count

{markdown_table(distribution_tables['Retry count'])}

### Customer segments

{markdown_table(distribution_tables['Customer segments'])}

### Recovery-success distribution

{markdown_table(distribution_tables['Recovery success'])}

Successful recoveries have a time-to-recovery range of {observed['time_to_recovery_hours'].min():.2f} to {observed['time_to_recovery_hours'].max():.2f} hours (median {observed['time_to_recovery_hours'].median():.2f}); failures correctly have no recovery-time value. The action mix is policy-driven, with {human_review_pct:.2f}% human review, which reflects the baseline policy rather than a natural production statistic.

## Observed action-conditional analysis

All rows are included below; cells absent from a grouping had no observed cases because the deterministic baseline did not select that action for that context.

### Failure reason × action

{markdown_table(failure_action)}

### Payment method × action

{markdown_table(method_action)}

### Amount band × action

{markdown_table(amount_action)}

### Retry count × action

{markdown_table(retry_action)}

### Customer segment × action

{markdown_table(segment_action)}

## Simulator-only potential-outcome analysis

The following values are simulator-generated potential outcomes, not observed real-world counterfactuals or causal effects.

### Mean potential outcomes by action

{markdown_table(mean_potential)}

### Mean potential success probability by failure reason × candidate action

{markdown_table(potential_by_reason)}

### Comparable feature groups: failure reason × amount band × retry count × candidate action

{markdown_table(comparable_potential)}

The candidate-action probability differences are material (for example, the global means differ by action). The within-failure optimal-action tables below show whether those interactions also change the preferred action across feature contexts.

## Task-difficulty audit: failure reason alone

### Observed recovery outcome by failure reason

{markdown_table(outcome_by_reason)}

### Probability-optimal action by failure reason

{markdown_table(best_by_reason)}

### Overall probability-optimal-action distribution

{markdown_table(optimal_distribution)}

### Context summaries where optimal action differs within the same failure reason

{markdown_table(context_summary.sort_values(['failure_reason', 'best_action_from_probability']))}

- Majority-class diagnostic accuracy for `recovery_success`: {difficulty['majority_accuracy']:.2f}%
- Failure-reason-only threshold rule accuracy for `recovery_success`: {difficulty['reason_rule_accuracy']:.2f}%
- Normalized outcome-entropy reduction from failure reason: {difficulty['entropy_reduction']:.2f}%
- Failure-reason modal lookup accuracy for probability-optimal action: {difficulty['best_action_lookup_accuracy']:.2f}%
- Failure reasons with multiple probability-optimal actions: {multiple_action_reason_count} of {len(best_by_reason)}

Interpretation: recovery success is **moderately learnable**, not deterministically fixed by failure reason. The action-selection target is assessed from the modal lookup accuracy and the within-reason variation above; no diagnostic model was trained.

## Counterfactual value and action regret

All five actions are candidate actions in this synthetic counterfactual environment. For each recovery case, **expected recovery value** is `potential_recovery_probability × amount_at_risk`. The probability-optimal candidate maximizes that value (equivalent to maximizing probability because every candidate shares the same amount at risk). **Absolute action regret** is the sum of optimal expected value minus the sum of expected value under the baseline-selected action. **Relative action regret** is absolute regret divided by optimal expected value.

- Baseline-selected expected recovered value: INR {expected_revenue:,.2f}
- Probability-optimal expected recovered value: INR {optimal_expected_revenue:,.2f}
- Probability-optimal simulated recovered value: INR {optimal_simulated_recovered_value:,.2f}
- Absolute action regret: INR {regret_absolute:,.2f}
- Relative action regret: {regret_relative * 100:.2f}%

## Held-out rule-based baseline audit

The baseline uses only t0 fields: route very-high-value (≥ INR {config.human_review_amount_threshold:,.0f}), defined ambiguous/repeated-risk, or policy-sensitive cases to `HUMAN_REVIEW`; block repeated actions during a cooldown; retry temporary failures only when retry count is below {config.max_retry_count}, amount is ≤ INR {config.automatic_action_amount_limit:,.0f}, and cooldown is eligible; use payment links/reminders for customer-action failures according to historical success, amount, and hour; route low-value/low-confidence or exhausted/blocked cases to `NO_ACTION`. It never reads potential or observed outcomes.

- Held-out test cases: {test_summary['cases']:,}
- Recovery rate: {test_summary['recovery_rate']:.2f}%
- Total recovered revenue: INR {test_summary['total_recovered_revenue']:,.2f}
- Average recovered amount, all test cases: INR {test_summary['average_recovered_amount_all_cases']:,.2f}
- Average recovered amount, recovered test cases: INR {test_summary['average_recovered_amount_successes']:,.2f}
- Expected recovered revenue: INR {test_summary['expected_recovered_revenue']:,.2f}
- Unrecovered-intervention-rate proxy: {test_summary['unrecovered_intervention_rate_proxy']:.2f}% (not a cost or proof of unnecessary contact)
- Policy violations: {test_summary['policy_violations']}

### Test: failure reason × observed action

{markdown_table(test_failure_action)}

### Test: observed action

{markdown_table(test_action)}

## Feature/outcome, leakage, and split audit

Pre-intervention prediction features are: `{', '.join(PREDICTION_FEATURE_COLUMNS)}`.

Post-intervention fields excluded from the feature matrix are `recovery_success`, `recovered_amount`, `time_to_recovery_hours`, future/post-action fields, and all `potential_*` counterfactual fields. The explicit feature-matrix check passed with {len(features.columns)} columns.

- Schema/leakage validation: passed.
- Historical snapshots strictly precede decision time: {(observed['last_history_event_at'] < observed['created_at']).all()}.
- Global chronological ordering: {observed['created_at'].is_monotonic_increasing}.
- No customer overlap across partitions: {no_customer_overlap}.
- Train: {config.train_start} to {config.train_end}; validation: {config.validation_start} to {config.validation_end}; test: {config.test_start} to {config.test_end}.

{markdown_table(split_table)}

No preprocessing is fitted or applied in this phase, so no training/validation/test preprocessing leakage exists. Future preprocessing must be fitted on the training partition only.

## Data-generation quality and recommendation

The simulator has a fixed seed, deterministic outputs, configurable thresholds, transparent probability logic, heterogeneous customer profiles, action-conditioned outcomes, non-independent features/outcomes, and valid chronology. It does not create a direct deterministic feature-to-success label; sampled outcomes remain stochastic.

{('Before ML, revise the action-probability mechanism to create meaningful within-failure action variation, then regenerate and repeat this audit.' if needs_revision else 'The simulator is suitable for the next, still pre-production ML preparation stage. Any later model must be evaluated against this baseline, the held-out test set, and the batch-level value/regret metrics above.')}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Audit complete: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Recommendation: {decision}")


if __name__ == "__main__":
    main()
