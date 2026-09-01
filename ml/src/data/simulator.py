"""Reproducible, action-conditional synthetic recovery-data simulator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from random import Random
from typing import Any

import pandas as pd

from .schemas import (
    CUSTOMER_ACTION_FAILURES,
    CUSTOMER_SEGMENTS,
    FAILURE_REASONS,
    PAYMENT_METHODS,
    RECOVERY_ACTIONS,
    RETRYABLE_FAILURES,
    REVIEW_FAILURES,
    SimulationConfig,
)


@dataclass
class SimulationResult:
    """Generated observed data, simulator-only potential outcomes, and customer master."""

    observed: pd.DataFrame
    potential_outcomes: pd.DataFrame
    customers: pd.DataFrame
    config: SimulationConfig


SEGMENT_WEIGHTS = (0.32, 0.33, 0.22, 0.08, 0.05)
METHOD_WEIGHTS = (0.45, 0.32, 0.15, 0.08)

# Each value is a synthetic starting probability before transparent adjustments.
ACTION_BASE_PROBABILITIES: dict[str, dict[str, float]] = {
    "BANK_TIMEOUT": {"RETRY": 0.72, "REMINDER": 0.40, "PAYMENT_LINK": 0.36, "HUMAN_REVIEW": 0.42, "NO_ACTION": 0.08},
    "GATEWAY_TIMEOUT": {"RETRY": 0.70, "REMINDER": 0.39, "PAYMENT_LINK": 0.35, "HUMAN_REVIEW": 0.41, "NO_ACTION": 0.08},
    "NETWORK_ERROR": {"RETRY": 0.68, "REMINDER": 0.38, "PAYMENT_LINK": 0.34, "HUMAN_REVIEW": 0.40, "NO_ACTION": 0.07},
    "TEMPORARY_BANK_ERROR": {"RETRY": 0.65, "REMINDER": 0.37, "PAYMENT_LINK": 0.33, "HUMAN_REVIEW": 0.40, "NO_ACTION": 0.07},
    "INSUFFICIENT_FUNDS": {"RETRY": 0.16, "REMINDER": 0.43, "PAYMENT_LINK": 0.50, "HUMAN_REVIEW": 0.32, "NO_ACTION": 0.10},
    "CARD_EXPIRED": {"RETRY": 0.04, "REMINDER": 0.27, "PAYMENT_LINK": 0.60, "HUMAN_REVIEW": 0.44, "NO_ACTION": 0.05},
    "INVALID_PAYMENT_DETAILS": {"RETRY": 0.05, "REMINDER": 0.30, "PAYMENT_LINK": 0.64, "HUMAN_REVIEW": 0.48, "NO_ACTION": 0.05},
    "CHECKOUT_ABANDONED": {"RETRY": 0.01, "REMINDER": 0.46, "PAYMENT_LINK": 0.55, "HUMAN_REVIEW": 0.28, "NO_ACTION": 0.08},
    "PAYMENT_METHOD_DECLINED": {"RETRY": 0.12, "REMINDER": 0.26, "PAYMENT_LINK": 0.40, "HUMAN_REVIEW": 0.45, "NO_ACTION": 0.08},
    "AMBIGUOUS_FAILURE": {"RETRY": 0.10, "REMINDER": 0.17, "PAYMENT_LINK": 0.25, "HUMAN_REVIEW": 0.42, "NO_ACTION": 0.05},
}


def _logit(probability: float) -> float:
    return math.log(probability / (1 - probability))


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def _weighted_choice(rng: Random, values: tuple[str, ...], weights: list[float] | tuple[float, ...]) -> str:
    return rng.choices(values, weights=weights, k=1)[0]


def _customer_profile(rng: Random, customer_id: str, cohort: str, split_start: datetime) -> dict[str, Any]:
    """Create correlated, non-sensitive customer behavior profiles."""
    segment = _weighted_choice(rng, CUSTOMER_SEGMENTS, SEGMENT_WEIGHTS)
    if segment == "RELIABLE":
        history_count, success_rate, average_amount = rng.randint(30, 120), rng.betavariate(18, 2), rng.lognormvariate(7.2, 0.55)
    elif segment == "NORMAL":
        history_count, success_rate, average_amount = rng.randint(12, 70), rng.betavariate(11, 4), rng.lognormvariate(7.0, 0.65)
    elif segment == "OCCASIONAL_FAILURE":
        history_count, success_rate, average_amount = rng.randint(8, 55), rng.betavariate(6, 5), rng.lognormvariate(6.8, 0.70)
    elif segment == "HIGH_VALUE":
        history_count, success_rate, average_amount = rng.randint(15, 85), rng.betavariate(13, 3), rng.lognormvariate(8.1, 0.70)
    else:
        history_count, success_rate, average_amount = rng.randint(2, 18), rng.betavariate(9, 4), rng.lognormvariate(6.7, 0.75)

    successful = round(history_count * success_rate)
    failed = history_count - successful
    created_at = split_start - timedelta(days=rng.randint(90, 1_100), hours=rng.randint(0, 23))
    lifetime_value = average_amount * successful * rng.uniform(0.9, 1.3)
    return {
        "customer_id": customer_id,
        "customer_created_at": created_at,
        "customer_segment": segment,
        "historical_transaction_count": history_count,
        "historical_success_count": successful,
        "historical_failure_count": failed,
        "historical_success_rate": successful / history_count,
        "historical_average_amount": round(average_amount, 2),
        "customer_lifetime_value": round(lifetime_value, 2),
        "simulation_cohort": cohort,
        "last_history_event_at": created_at + timedelta(days=rng.randint(1, 60)),
    }


def _random_timestamp(rng: Random, start: datetime, end: datetime) -> datetime:
    return start + timedelta(seconds=rng.randint(0, int((end - start).total_seconds())))


def _failure_reason(rng: Random, method: str, customer: dict[str, Any], retry_count: int) -> str:
    """Sample failure reasons conditionally on method, history, and repeated attempts."""
    weights = {reason: 1.0 for reason in FAILURE_REASONS}
    if method == "UPI":
        for reason in ("BANK_TIMEOUT", "NETWORK_ERROR", "TEMPORARY_BANK_ERROR"):
            weights[reason] *= 1.8
    elif method == "CARD":
        for reason in ("CARD_EXPIRED", "INVALID_PAYMENT_DETAILS", "PAYMENT_METHOD_DECLINED"):
            weights[reason] *= 1.9
    elif method == "NETBANKING":
        for reason in ("BANK_TIMEOUT", "GATEWAY_TIMEOUT", "INSUFFICIENT_FUNDS"):
            weights[reason] *= 1.45
    else:
        weights["CHECKOUT_ABANDONED"] *= 1.8

    failure_rate = 1.0 - float(customer["historical_success_rate"])
    weights["INSUFFICIENT_FUNDS"] *= 1 + failure_rate * 2.0
    weights["PAYMENT_METHOD_DECLINED"] *= 1 + failure_rate
    weights["AMBIGUOUS_FAILURE"] *= 1 + retry_count * 0.45
    return _weighted_choice(rng, FAILURE_REASONS, [weights[reason] for reason in FAILURE_REASONS])


def _amount(rng: Random, customer: dict[str, Any]) -> float:
    """Use a customer-conditioned log-normal long-tail INR amount distribution."""
    multiplier = rng.lognormvariate(0.0, 0.72)
    value = float(customer["historical_average_amount"]) * multiplier
    return round(min(150_000.0, max(100.0, value)), 2)


def _choose_action(row: dict[str, Any], config: SimulationConfig) -> str:
    """Transparent baseline policy; this is not a production policy engine."""
    reason = str(row["failure_reason"])
    if bool(row["human_review_required"]):
        return "HUMAN_REVIEW"
    if int(row["retry_count"]) > 0 and not bool(row["cooldown_eligible"]):
        return "NO_ACTION"
    if (
        reason in RETRYABLE_FAILURES
        and int(row["retry_count"]) < config.max_retry_count
        and bool(row["amount_limit_eligible"])
        and bool(row["cooldown_eligible"])
    ):
        return "RETRY"
    if reason in ("CARD_EXPIRED", "INVALID_PAYMENT_DETAILS"):
        return "PAYMENT_LINK" if float(row["amount_at_risk"]) <= config.automatic_action_amount_limit else "NO_ACTION"
    if reason == "INSUFFICIENT_FUNDS":
        if float(row["amount_at_risk"]) < config.low_value_no_action_threshold and float(row["historical_success_rate"]) < 0.55:
            return "NO_ACTION"
        return "REMINDER" if float(row["historical_success_rate"]) >= 0.72 and float(row["amount_at_risk"]) <= 10_000 else "PAYMENT_LINK"
    if reason == "CHECKOUT_ABANDONED":
        if float(row["amount_at_risk"]) < config.low_value_no_action_threshold and int(row["retry_count"]) > 0:
            return "NO_ACTION"
        return "REMINDER" if 8 <= int(row["hour_of_day"]) < 20 else "PAYMENT_LINK"
    if reason == "PAYMENT_METHOD_DECLINED":
        return "PAYMENT_LINK" if float(row["historical_success_rate"]) >= 0.65 and int(row["retry_count"]) < config.max_retry_count else "NO_ACTION"
    if reason == "AMBIGUOUS_FAILURE":
        return "NO_ACTION"
    return "NO_ACTION"


def _recovery_probability(row: dict[str, Any], action: str) -> float:
    """Interpretable v2 logit model with action-specific pre-intervention effects.

    Effects intentionally change relative action rankings for comparable failure
    reasons. They are synthetic assumptions, not production-payment statistics.
    """
    reason = str(row["failure_reason"])
    method = str(row["payment_method"])
    segment = str(row["customer_segment"])
    history = float(row["historical_success_rate"]) - 0.70
    retry_count = min(int(row["retry_count"]), 3)
    failure_pressure = min(int(row["previous_failure_count"]), 20) / 20
    amount = float(row["amount_at_risk"])
    medium_amount = 5_000 < amount <= 25_000
    high_amount = amount > 25_000
    off_hours = int(row["hour_of_day"]) < 6 or int(row["hour_of_day"]) >= 22
    log_odds = _logit(ACTION_BASE_PROBABILITIES[reason][action])

    # Customer history × retry, amount × action, and retry-count × retry.
    if action == "RETRY":
        log_odds += 1.20 * history - 0.72 * retry_count - 0.45 * failure_pressure
        log_odds += -0.28 * medium_amount - 0.95 * high_amount
        log_odds += 0.20 if method in ("UPI", "NETBANKING") else -0.12
        log_odds += 0.18 if segment == "RELIABLE" else (-0.20 if segment == "OCCASIONAL_FAILURE" else 0.0)
        log_odds += -0.12 if off_hours else 0.0
    elif action == "REMINDER":
        log_odds += 0.52 * history - 0.20 * retry_count - 0.16 * failure_pressure
        log_odds += 0.12 * (not high_amount) - 0.32 * high_amount
        log_odds += 0.18 if method == "UPI" else (0.12 if method == "WALLET" else 0.0)
        log_odds += 0.16 if segment in ("RELIABLE", "LOW_FREQUENCY") else 0.0
        log_odds += -0.24 if off_hours else 0.10
    elif action == "PAYMENT_LINK":
        log_odds += 0.42 * history - 0.24 * retry_count - 0.22 * failure_pressure
        log_odds += 0.10 * medium_amount - 0.42 * high_amount
        log_odds += 0.30 if method == "CARD" else (-0.12 if method == "UPI" else 0.0)
        log_odds += 0.18 if segment == "HIGH_VALUE" else 0.0
        log_odds += -0.16 if off_hours else 0.0
    elif action == "HUMAN_REVIEW":
        log_odds += -0.36 * history + 0.68 * retry_count + 0.72 * failure_pressure
        log_odds += 0.32 * medium_amount + 1.18 * high_amount
        log_odds += 0.16 if method == "CARD" else 0.0
        log_odds += 0.24 if segment in ("OCCASIONAL_FAILURE", "HIGH_VALUE") else 0.0
        log_odds += 0.08 if off_hours else 0.0
    else:  # NO_ACTION represents spontaneous recovery without intervention.
        log_odds += -0.58 * history - 0.10 * retry_count - 0.08 * failure_pressure
        log_odds += -0.35 * high_amount - 0.10 * medium_amount
        log_odds += 0.10 if off_hours else 0.0

    # Failure type × action interactions; these alter rankings in feature context.
    if reason in RETRYABLE_FAILURES and action == "HUMAN_REVIEW":
        log_odds += 0.52 * high_amount + 0.28 * retry_count + 0.22 * failure_pressure
    if reason in RETRYABLE_FAILURES and action == "RETRY":
        log_odds += 0.25 * (history > 0.08) - 0.38 * high_amount
    if reason == "INSUFFICIENT_FUNDS":
        if action == "REMINDER":
            log_odds += 0.48 * (history > 0.05) + 0.20 * (amount <= 10_000)
        elif action == "PAYMENT_LINK":
            log_odds += 0.34 * (method == "CARD") + 0.24 * medium_amount - 0.26 * failure_pressure
        elif action == "HUMAN_REVIEW":
            log_odds += 0.62 * high_amount + 0.38 * failure_pressure
    if reason in ("CARD_EXPIRED", "INVALID_PAYMENT_DETAILS"):
        if action == "PAYMENT_LINK":
            log_odds += 0.36 * (method == "CARD") + 0.20 * history - 0.35 * high_amount
        elif action == "HUMAN_REVIEW":
            log_odds += 0.68 * high_amount + 0.34 * retry_count
    if reason == "CHECKOUT_ABANDONED":
        if action == "REMINDER":
            log_odds += 0.52 * (not off_hours) + 0.28 * history
        elif action == "PAYMENT_LINK":
            log_odds += 0.30 * (method == "CARD") + 0.22 * off_hours
        elif action == "HUMAN_REVIEW":
            log_odds += 0.76 * high_amount + 0.35 * failure_pressure
    if reason in REVIEW_FAILURES and action == "HUMAN_REVIEW":
        log_odds += 0.44 * failure_pressure + 0.45 * high_amount
    if reason == "PAYMENT_METHOD_DECLINED" and action == "PAYMENT_LINK":
        log_odds += 0.42 * (history > 0.0) + 0.26 * (method == "CARD")
    if reason == "AMBIGUOUS_FAILURE" and action == "PAYMENT_LINK":
        log_odds += 0.38 * history - 0.30 * failure_pressure

    return round(max(0.01, min(0.95, _sigmoid(log_odds))), 4)


def _time_to_recovery(rng: Random, action: str) -> float:
    """Action-conditioned recovery delay in hours, used only for successful outcomes."""
    ranges = {
        "RETRY": (0.05, 2.0),
        "REMINDER": (4.0, 72.0),
        "PAYMENT_LINK": (2.0, 96.0),
        "HUMAN_REVIEW": (12.0, 120.0),
        "NO_ACTION": (24.0, 168.0),
    }
    low, high = ranges[action]
    return round(rng.uniform(low, high), 2)


def _split_plan(config: SimulationConfig) -> tuple[tuple[str, int, datetime, datetime], ...]:
    if config.train_cases + config.validation_cases + config.test_cases != config.num_cases:
        raise ValueError("Split case counts must sum to num_cases.")
    return (
        ("train", config.train_cases, datetime.fromisoformat(config.train_start), datetime.fromisoformat(config.train_end)),
        ("validation", config.validation_cases, datetime.fromisoformat(config.validation_start), datetime.fromisoformat(config.validation_end)),
        ("test", config.test_cases, datetime.fromisoformat(config.test_start), datetime.fromisoformat(config.test_end)),
    )


def generate_recovery_data(config: SimulationConfig | None = None) -> SimulationResult:
    """Generate a deterministic, chronological synthetic recovery-opportunity cohort."""
    config = config or SimulationConfig()
    if config.num_customers < 3:
        raise ValueError("At least three customers are required for split-safe generation.")
    rng = Random(config.seed)
    plan = _split_plan(config)
    customer_counts = (int(config.num_customers * 0.70), int(config.num_customers * 0.15), config.num_customers - int(config.num_customers * 0.85))
    customers: dict[str, dict[str, Any]] = {}
    customer_master_rows: list[dict[str, Any]] = []
    customer_pools: dict[str, list[str]] = {}
    cursor = 1
    for (split, _, start, _), count in zip(plan, customer_counts, strict=True):
        pool: list[str] = []
        for _ in range(count):
            customer_id = f"CUST-{cursor:05d}"
            customers[customer_id] = _customer_profile(rng, customer_id, split, start)
            customer_master_rows.append(
                {key: value for key, value in customers[customer_id].items() if key != "last_history_event_at"}
            )
            pool.append(customer_id)
            cursor += 1
        customer_pools[split] = pool

    skeletons: list[dict[str, Any]] = []
    payment_number = 1
    for split, count, start, end in plan:
        pool = customer_pools[split]
        selected = pool.copy()  # ensure every split customer appears at least once
        while len(selected) < count:
            selected.append(rng.choice(pool))
        rng.shuffle(selected)
        for customer_id in selected[:count]:
            skeletons.append(
                {
                    "customer_id": customer_id,
                    "created_at": _random_timestamp(rng, start, end),
                    "split": split,
                    "payment_id": f"PAY-{payment_number:06d}",
                    "recovery_case_id": f"RC-{payment_number:06d}",
                }
            )
            payment_number += 1
    skeletons.sort(key=lambda item: (item["created_at"], item["payment_id"]))

    observed_rows: list[dict[str, Any]] = []
    potential_rows: list[dict[str, Any]] = []
    for skeleton in skeletons:
        customer = customers[str(skeleton["customer_id"])]
        created_at = skeleton["created_at"]
        last_event = customer["last_history_event_at"]
        if last_event >= created_at:
            last_event = created_at - timedelta(hours=rng.uniform(1, 72))
        time_since_hours = max(1.0, (created_at - last_event).total_seconds() / 3600)
        transaction_count = int(customer["historical_transaction_count"])
        success_count = int(customer["historical_success_count"])
        failure_count = int(customer["historical_failure_count"])
        success_rate = success_count / max(1, transaction_count)
        frequency = transaction_count / max(1.0, (created_at - customer["customer_created_at"]).days / 30)
        retry_count = 0 if rng.random() > min(0.42, 0.07 + failure_count / max(1, transaction_count) * 0.35) else rng.choices((1, 2, 3), (0.67, 0.25, 0.08), k=1)[0]
        method = _weighted_choice(rng, PAYMENT_METHODS, METHOD_WEIGHTS)
        amount = _amount(rng, customer)
        row: dict[str, Any] = {
            "recovery_case_id": skeleton["recovery_case_id"],
            "payment_id": skeleton["payment_id"],
            "customer_id": skeleton["customer_id"],
            "customer_created_at": customer["customer_created_at"],
            "customer_segment": customer["customer_segment"],
            "created_at": created_at,
            "amount": amount,
            "amount_at_risk": amount,
            "currency": "INR",
            "payment_method": method,
            "payment_status": "FAILED",
            "attempt_count": retry_count + 1,
            "retry_count": retry_count,
            "previous_failure_count": failure_count,
            "previous_success_count": success_count,
            "historical_transaction_count": transaction_count,
            "historical_success_count": success_count,
            "historical_failure_count": failure_count,
            "historical_success_rate": round(success_rate, 4),
            "historical_average_amount": customer["historical_average_amount"],
            "customer_lifetime_value": customer["customer_lifetime_value"],
            "time_since_previous_payment_hours": round(time_since_hours, 2),
            "customer_transaction_frequency": round(frequency, 4),
            "hour_of_day": created_at.hour,
            "day_of_week": created_at.weekday(),
            "last_history_event_at": last_event,
            "cooldown_eligible": time_since_hours >= config.cooldown_hours or retry_count == 0,
            "amount_limit_eligible": amount <= config.automatic_action_amount_limit,
            "human_review_required": False,
            "split": skeleton["split"],
        }
        row["failure_reason"] = _failure_reason(rng, method, row, retry_count)
        if row["failure_reason"] == "CHECKOUT_ABANDONED":
            row["payment_status"] = "ABANDONED"
        row["human_review_required"] = (
            amount >= config.human_review_amount_threshold
            or (row["failure_reason"] == "AMBIGUOUS_FAILURE" and (retry_count >= 1 or failure_count >= config.human_review_previous_failures_threshold or success_rate < 0.60))
            or (retry_count >= config.human_review_retry_count_threshold and amount >= 15_000)
            or (row["failure_reason"] == "PAYMENT_METHOD_DECLINED" and retry_count >= config.max_retry_count and success_rate < 0.55)
        )
        row["observed_action"] = _choose_action(row, config)
        row["action_allowed"] = row["observed_action"] != "NO_ACTION"

        selected_outcome: dict[str, Any] | None = None
        for action in RECOVERY_ACTIONS:
            probability = _recovery_probability(row, action)
            success = int(rng.random() < probability)
            potential = {
                "recovery_case_id": row["recovery_case_id"],
                "payment_id": row["payment_id"],
                "customer_id": row["customer_id"],
                "potential_action": action,
                "potential_recovery_probability": probability,
                "potential_recovery_success": success,
                "potential_recovered_amount": amount if success else 0.0,
                "potential_time_to_recovery_hours": _time_to_recovery(rng, action) if success else None,
                "ground_truth_type": "SIMULATOR_ONLY_COUNTERFACTUAL",
            }
            potential_rows.append(potential)
            if action == row["observed_action"]:
                selected_outcome = potential
        if selected_outcome is None:
            raise RuntimeError("Observed action did not receive a potential outcome.")
        row["recovery_success"] = selected_outcome["potential_recovery_success"]
        row["recovered_amount"] = selected_outcome["potential_recovered_amount"]
        row["time_to_recovery_hours"] = selected_outcome["potential_time_to_recovery_hours"]
        observed_rows.append(row)

        # Future snapshots see this failed payment, but never this row's recovery outcome.
        customer["historical_transaction_count"] = transaction_count + 1
        customer["historical_failure_count"] = failure_count + 1
        customer["last_history_event_at"] = created_at

    observed = pd.DataFrame(observed_rows).sort_values(["created_at", "payment_id"]).reset_index(drop=True)
    potential = pd.DataFrame(potential_rows).sort_values(["recovery_case_id", "potential_action"]).reset_index(drop=True)
    customer_frame = pd.DataFrame(customer_master_rows).sort_values("customer_id").reset_index(drop=True)
    return SimulationResult(observed=observed, potential_outcomes=potential, customers=customer_frame, config=config)
