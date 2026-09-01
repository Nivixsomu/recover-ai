"""Generate, validate, partition, and report deterministic synthetic recovery data."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.data.simulator import SimulationConfig, generate_recovery_data
from ml.src.data.validation import validate_simulation


PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
FINAL_DIR = PROJECT_ROOT / "ml" / "data" / "final"
DOCS_DIR = PROJECT_ROOT / "docs"


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """Render a small Markdown table without an optional third-party package."""
    selected = frame.loc[:, columns].copy()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for values in selected.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join([header, divider, *rows])


def _write_report(observed: pd.DataFrame, config: SimulationConfig) -> None:
    failure = observed["failure_reason"].value_counts().rename_axis("failure_reason").reset_index(name="cases")
    failure["percentage"] = (failure["cases"] / len(observed) * 100).round(2)
    actions = observed["observed_action"].value_counts().rename_axis("observed_action").reset_index(name="cases")
    actions["percentage"] = (actions["cases"] / len(observed) * 100).round(2)
    recovery_by_failure = observed.groupby("failure_reason", as_index=False).agg(cases=("recovery_case_id", "size"), recovery_rate=("recovery_success", "mean"), recovered_revenue=("recovered_amount", "sum"))
    recovery_by_failure["recovery_rate"] = (recovery_by_failure["recovery_rate"] * 100).round(2)
    recovery_by_action = observed.groupby("observed_action", as_index=False).agg(cases=("recovery_case_id", "size"), recovery_rate=("recovery_success", "mean"), recovered_revenue=("recovered_amount", "sum"))
    recovery_by_action["recovery_rate"] = (recovery_by_action["recovery_rate"] * 100).round(2)
    amount = observed["amount_at_risk"].describe().round(2)
    missing = observed.drop(columns=["time_to_recovery_hours"]).isna().sum()
    missing = missing[missing > 0].rename_axis("column").reset_index(name="missing_values")
    split_stats = observed.groupby("split", as_index=False).agg(cases=("recovery_case_id", "size"), customers=("customer_id", "nunique"), earliest=("created_at", "min"), latest=("created_at", "max"))
    report = f"""# Generated Synthetic Recovery Data Report

This report describes a controlled synthetic experimental dataset. It is **not evidence of Razorpay, bank, merchant, or customer production performance**. The UCI Online Retail archive was not read by this generator; it remains an optional behavioral reference only. UCI Credit Card Default was not used.

## Reproducibility

- Simulator: `{config.version}`
- Random seed: `{config.seed}`
- Python generation command: `python scripts/generate_recovery_data.py`
- Opportunities/payments: {len(observed):,}
- Customers: {observed['customer_id'].nunique():,}
- Currency: INR
- Potential-outcome rows: {len(observed) * 5:,} (simulator-only counterfactual ground truth)

## Baseline-policy configuration

- Maximum retry count before review: {config.max_retry_count}
- Autonomous action amount limit: INR {config.automatic_action_amount_limit:,.0f}
- Human-review amount threshold: INR {config.human_review_amount_threshold:,.0f}
- Human-review retry-count threshold: {config.human_review_retry_count_threshold}
- Human-review historical-failure threshold: {config.human_review_previous_failures_threshold}
- Low-value no-action threshold: INR {config.low_value_no_action_threshold:,.0f}
- Cooldown: {config.cooldown_hours:.0f} hours since prior payment, unless this is the first retry

## Overall outcomes

- Recovery rate: **{observed['recovery_success'].mean() * 100:.2f}%**
- Total simulated recovered revenue: **INR {observed['recovered_amount'].sum():,.2f}**
- Total amount at risk: INR {observed['amount_at_risk'].sum():,.2f}
- Mean amount at risk: INR {observed['amount_at_risk'].mean():,.2f}

## Failure distribution

{_markdown_table(failure, ['failure_reason', 'cases', 'percentage'])}

## Payment-method distribution

{_markdown_table(observed['payment_method'].value_counts().rename_axis('payment_method').reset_index(name='cases'), ['payment_method', 'cases'])}

## Action distribution

{_markdown_table(actions, ['observed_action', 'cases', 'percentage'])}

## Amount statistics (INR)

```text
{amount.to_string()}
```

## Customer behavior at decision time

- Mean historical transaction count: {observed['historical_transaction_count'].mean():.2f}
- Mean historical success rate: {observed['historical_success_rate'].mean() * 100:.2f}%
- Mean customer lifetime value: INR {observed['customer_lifetime_value'].mean():,.2f}
- Customer segments: {observed['customer_segment'].value_counts().to_dict()}

## Recovery outcomes by failure type

{_markdown_table(recovery_by_failure.sort_values('failure_reason'), ['failure_reason', 'cases', 'recovery_rate', 'recovered_revenue'])}

## Recovery outcomes by observed action

{_markdown_table(recovery_by_action.sort_values('observed_action'), ['observed_action', 'cases', 'recovery_rate', 'recovered_revenue'])}

## Leakage-safe partitions

{_markdown_table(split_stats, ['split', 'cases', 'customers', 'earliest', 'latest'])}

Partition boundaries are train: {config.train_start} through {config.train_end}; validation: {config.validation_start} through {config.validation_end}; test: {config.test_start} through {config.test_end}. Customers are assigned to exactly one partition. Feature snapshots use history strictly before each `created_at`; outcome and potential-outcome columns are excluded from the explicit prediction feature matrix.

## Data quality and validation

- Duplicate payment IDs: {observed['payment_id'].duplicated().sum()}
- Duplicate recovery case IDs: {observed['recovery_case_id'].duplicated().sum()}
- Missing values excluding expected null `time_to_recovery_hours` on failed recoveries: {missing.to_dict(orient='records') if not missing.empty else 'None'}
- Expected null `time_to_recovery_hours` values for unsuccessful recoveries: {(observed['recovery_success'] == 0).sum()}
- Validation: **passed** — identifiers, references, timestamps, categories, amounts, outcomes, potential-outcome linkage, chronology, partition boundaries, customer separation, and leakage-column checks all passed.
"""
    (DOCS_DIR / "generated_data_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    config = SimulationConfig()
    result = generate_recovery_data(config)
    validate_simulation(result.observed, result.potential_outcomes, result.customers, config)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    result.observed.to_csv(PROCESSED_DIR / "recovery_observed.csv", index=False)
    result.potential_outcomes.to_csv(PROCESSED_DIR / "recovery_potential_outcomes.csv", index=False)
    result.customers.to_csv(PROCESSED_DIR / "recovery_customers.csv", index=False)
    for split, filename in (("train", "recovery_train.csv"), ("validation", "recovery_validation.csv"), ("test", "recovery_test.csv")):
        result.observed.loc[result.observed["split"] == split].to_csv(FINAL_DIR / filename, index=False)
    metadata = {"simulator": config.version, "seed": config.seed, "config": config.as_dict(), "input_datasets": "None; controlled synthetic generation only"}
    (FINAL_DIR / "recovery_generation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_report(result.observed, config)
    print(f"Generated {len(result.observed):,} recovery opportunities with seed {config.seed}.")
    print("Validation passed. Outputs written to ml/data/processed and ml/data/final.")


if __name__ == "__main__":
    main()
