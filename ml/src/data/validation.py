"""Schema, consistency, split, and leakage validation for simulated recovery data."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .schemas import FAILURE_REASONS, PAYMENT_METHODS, RECOVERY_ACTIONS, SimulationConfig


PREDICTION_FEATURE_COLUMNS = (
    "amount",
    "payment_method",
    "failure_reason",
    "attempt_count",
    "retry_count",
    "previous_failure_count",
    "previous_success_count",
    "historical_success_rate",
    "historical_average_amount",
    "customer_lifetime_value",
    "time_since_previous_payment_hours",
    "customer_transaction_frequency",
    "hour_of_day",
    "day_of_week",
    "cooldown_eligible",
    "amount_limit_eligible",
    "human_review_required",
    "action_allowed",
)
LEAKAGE_COLUMNS = {
    "recovery_success",
    "recovered_amount",
    "time_to_recovery_hours",
    "recovery_outcome_at",
    "future_payment_outcomes",
    "post_intervention_information",
    "potential_recovery_probability",
    "potential_recovery_success",
    "potential_recovered_amount",
    "potential_time_to_recovery_hours",
}


def prediction_feature_matrix(observed: pd.DataFrame) -> pd.DataFrame:
    """Return the explicit pre-intervention feature matrix only."""
    validate_no_leakage_columns(PREDICTION_FEATURE_COLUMNS)
    return observed.loc[:, list(PREDICTION_FEATURE_COLUMNS)].copy()


def validate_no_leakage_columns(columns: Iterable[str]) -> None:
    leaked = set(columns).intersection(LEAKAGE_COLUMNS)
    if leaked:
        raise ValueError(f"Outcome or post-intervention columns in feature matrix: {sorted(leaked)}")


def validate_simulation(
    observed: pd.DataFrame,
    potential: pd.DataFrame,
    customers: pd.DataFrame,
    config: SimulationConfig,
) -> None:
    """Fail loudly if identifiers, outcomes, chronology, partitions, or leakage fail."""
    required = {
        "recovery_case_id", "payment_id", "customer_id", "created_at", "amount", "amount_at_risk",
        "currency", "payment_method", "payment_status", "failure_reason", "observed_action",
        "recovery_success", "recovered_amount", "time_to_recovery_hours", "last_history_event_at", "split",
    }
    missing = required.difference(observed.columns)
    if missing:
        raise ValueError(f"Observed dataset missing required columns: {sorted(missing)}")
    if not observed["payment_id"].is_unique or not observed["recovery_case_id"].is_unique:
        raise ValueError("Payment IDs and recovery case IDs must be unique.")
    if not set(observed["customer_id"]).issubset(set(customers["customer_id"])):
        raise ValueError("Observed rows contain an invalid customer reference.")
    created_at = pd.to_datetime(observed["created_at"], errors="coerce")
    history_at = pd.to_datetime(observed["last_history_event_at"], errors="coerce")
    if created_at.isna().any() or history_at.isna().any() or not (history_at < created_at).all():
        raise ValueError("Invalid timestamps or historical snapshot not strictly before decision time.")
    if not created_at.is_monotonic_increasing:
        raise ValueError("Observed payment events must be chronological.")
    if (observed["amount"] <= 0).any() or (observed["amount_at_risk"] <= 0).any():
        raise ValueError("Amounts must be positive.")
    if not (observed["currency"] == "INR").all():
        raise ValueError("Currency must be INR.")
    if not set(observed["payment_method"]).issubset(PAYMENT_METHODS):
        raise ValueError("Invalid payment method found.")
    if not set(observed["failure_reason"]).issubset(FAILURE_REASONS):
        raise ValueError("Invalid failure reason found.")
    if not set(observed["observed_action"]).issubset(RECOVERY_ACTIONS):
        raise ValueError("Invalid recovery action found.")
    if not set(observed["recovery_success"]).issubset({0, 1}):
        raise ValueError("recovery_success must be binary.")
    if (observed["recovered_amount"] < 0).any() or (observed["recovered_amount"] > observed["amount_at_risk"]).any():
        raise ValueError("Recovered amount violates amount-at-risk bounds.")
    failures = observed["recovery_success"] == 0
    successes = observed["recovery_success"] == 1
    if (observed.loc[failures, "recovered_amount"] != 0).any() or observed.loc[successes, "time_to_recovery_hours"].isna().any():
        raise ValueError("Recovery outcome and amount/time consistency failed.")
    if observed.loc[failures, "time_to_recovery_hours"].notna().any():
        raise ValueError("Time to recovery must only exist for successful recovery.")
    validate_no_leakage_columns(PREDICTION_FEATURE_COLUMNS)
    _validate_partitions(observed, config)
    _validate_potential_outcomes(observed, potential)


def _validate_partitions(observed: pd.DataFrame, config: SimulationConfig) -> None:
    boundaries = {
        "train": (pd.Timestamp(config.train_start), pd.Timestamp(config.train_end), config.train_cases),
        "validation": (pd.Timestamp(config.validation_start), pd.Timestamp(config.validation_end), config.validation_cases),
        "test": (pd.Timestamp(config.test_start), pd.Timestamp(config.test_end), config.test_cases),
    }
    all_customers: list[set[str]] = []
    for split, (start, end, expected_count) in boundaries.items():
        frame = observed.loc[observed["split"] == split]
        dates = pd.to_datetime(frame["created_at"])
        if len(frame) != expected_count or not dates.between(start, end).all():
            raise ValueError(f"Invalid {split} partition size or time boundary.")
        all_customers.append(set(frame["customer_id"]))
    if any(all_customers[left].intersection(all_customers[right]) for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("Customers must not overlap across train/validation/test partitions.")


def _validate_potential_outcomes(observed: pd.DataFrame, potential: pd.DataFrame) -> None:
    if len(potential) != len(observed) * len(RECOVERY_ACTIONS):
        raise ValueError("Potential-outcomes dataset must contain one row per action per recovery case.")
    if not set(potential["potential_action"]).issubset(RECOVERY_ACTIONS):
        raise ValueError("Potential-outcomes dataset contains an invalid action.")
    actions_per_case = potential.groupby("recovery_case_id")["potential_action"].nunique()
    if not (actions_per_case == len(RECOVERY_ACTIONS)).all():
        raise ValueError("Each recovery case must include every potential action exactly once.")
    selected = potential.merge(
        observed[["recovery_case_id", "observed_action", "recovery_success", "recovered_amount", "time_to_recovery_hours"]],
        left_on=["recovery_case_id", "potential_action"],
        right_on=["recovery_case_id", "observed_action"],
        how="inner",
    )
    if len(selected) != len(observed):
        raise ValueError("Every observed action must map to exactly one potential outcome.")
    if not (selected["potential_recovery_success"] == selected["recovery_success"]).all():
        raise ValueError("Observed outcome must match the selected potential outcome.")
    if not (selected["potential_recovered_amount"] == selected["recovered_amount"]).all():
        raise ValueError("Observed recovered amount must match selected potential outcome.")
