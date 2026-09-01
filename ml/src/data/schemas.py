"""Shared simulation constants and configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass


PAYMENT_METHODS = ("UPI", "CARD", "NETBANKING", "WALLET")
RETRYABLE_FAILURES = (
    "BANK_TIMEOUT",
    "GATEWAY_TIMEOUT",
    "NETWORK_ERROR",
    "TEMPORARY_BANK_ERROR",
)
CUSTOMER_ACTION_FAILURES = (
    "INSUFFICIENT_FUNDS",
    "CARD_EXPIRED",
    "INVALID_PAYMENT_DETAILS",
    "CHECKOUT_ABANDONED",
)
REVIEW_FAILURES = ("PAYMENT_METHOD_DECLINED", "AMBIGUOUS_FAILURE")
FAILURE_REASONS = RETRYABLE_FAILURES + CUSTOMER_ACTION_FAILURES + REVIEW_FAILURES
RECOVERY_ACTIONS = ("RETRY", "REMINDER", "PAYMENT_LINK", "HUMAN_REVIEW", "NO_ACTION")
CUSTOMER_SEGMENTS = (
    "RELIABLE",
    "NORMAL",
    "OCCASIONAL_FAILURE",
    "HIGH_VALUE",
    "LOW_FREQUENCY",
)


@dataclass(frozen=True)
class SimulationConfig:
    """All deterministic simulation and baseline-policy settings."""

    seed: int = 20260401
    num_cases: int = 20_000
    num_customers: int = 4_000
    train_cases: int = 14_000
    validation_cases: int = 3_000
    test_cases: int = 3_000
    train_start: str = "2025-01-01T00:00:00"
    train_end: str = "2025-09-30T23:59:59"
    validation_start: str = "2025-10-01T00:00:00"
    validation_end: str = "2025-11-15T23:59:59"
    test_start: str = "2025-11-16T00:00:00"
    test_end: str = "2025-12-31T23:59:59"
    max_retry_count: int = 2
    automatic_action_amount_limit: float = 25_000.0
    human_review_amount_threshold: float = 50_000.0
    human_review_retry_count_threshold: int = 3
    human_review_previous_failures_threshold: int = 20
    low_value_no_action_threshold: float = 300.0
    cooldown_hours: float = 24.0
    version: str = "recovery-simulator-v2"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
