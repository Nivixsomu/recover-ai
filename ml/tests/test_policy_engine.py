"""Tests for PolicyEngine safety constraints, blocking rules, and fallbacks."""

import pytest

from ml.src.models.action_ranker import ActionCandidateScore, ActionRankingResult
from ml.src.policy import PolicyConfig, PolicyEngine, PolicyEvaluationResult


@pytest.fixture
def policy_engine() -> PolicyEngine:
    return PolicyEngine(PolicyConfig())


def make_mock_ranking(
    top_action: str = "RETRY",
    amount: float = 1000.0,
    actions_order: list[str] | None = None,
) -> ActionRankingResult:
    if actions_order is None:
        actions_order = ["RETRY", "PAYMENT_LINK", "REMINDER", "HUMAN_REVIEW", "NO_ACTION"]

    scores = [
        ActionCandidateScore(
            action=act,
            predicted_probability=0.8 - idx * 0.1,
            expected_recovery_value=(0.8 - idx * 0.1) * amount,
            rank=idx + 1,
        )
        for idx, act in enumerate(actions_order)
    ]
    return ActionRankingResult(
        amount_at_risk=amount,
        top_action=scores[0].action,
        top_probability=scores[0].predicted_probability,
        top_expected_value=scores[0].expected_recovery_value,
        rankings=scores,
        case_metadata={},
    )


def test_retry_limit_blocks_retry(policy_engine: PolicyEngine):
    """Verify retry_count >= max_automatic_retries blocks RETRY and triggers fallback."""
    case = {
        "amount_at_risk": 1500.0,
        "retry_count": 2,  # Limit is 2
        "failure_reason": "BANK_TIMEOUT",
        "cooldown_eligible": True,
    }
    ranking = make_mock_ranking(top_action="RETRY", amount=1500.0)
    decision = policy_engine.evaluate_ranking(case, ranking)

    assert "RETRY" in decision.blocked_actions
    assert "ERR_RETRY_LIMIT_EXCEEDED" in decision.blocked_actions["RETRY"]
    assert decision.fallback_occurred is True
    assert decision.selected_action == "PAYMENT_LINK"  # Rank 2 fallback


def test_cooldown_blocks_retry_and_reminder(policy_engine: PolicyEngine):
    """Verify active cooldown blocks RETRY and REMINDER."""
    case = {
        "amount_at_risk": 1000.0,
        "retry_count": 1,
        "cooldown_eligible": False,
        "failure_reason": "NETWORK_ERROR",
    }
    ranking = make_mock_ranking(actions_order=["RETRY", "REMINDER", "PAYMENT_LINK", "NO_ACTION", "HUMAN_REVIEW"])
    decision = policy_engine.evaluate_ranking(case, ranking)

    assert "RETRY" in decision.blocked_actions
    assert "REMINDER" in decision.blocked_actions
    assert decision.selected_action == "PAYMENT_LINK"


def test_amount_limit_blocks_automatic_retry(policy_engine: PolicyEngine):
    """Verify amount > 25,000 blocks automatic RETRY."""
    case = {
        "amount_at_risk": 35000.0,
        "retry_count": 0,
        "cooldown_eligible": True,
        "failure_reason": "BANK_TIMEOUT",
    }
    ranking = make_mock_ranking(top_action="RETRY", amount=35000.0)
    decision = policy_engine.evaluate_ranking(case, ranking)

    assert "RETRY" in decision.blocked_actions
    assert "ERR_AMOUNT_EXCEEDS_AUTONOMY_LIMIT" in decision.blocked_actions["RETRY"]


def test_human_review_threshold_blocks_and_falls_back(policy_engine: PolicyEngine):
    """Verify low-value ticket without escalation criteria blocks HUMAN_REVIEW."""
    case = {
        "amount_at_risk": 500.0,
        "retry_count": 0,
        "previous_failure_count": 1,
        "failure_reason": "INSUFFICIENT_FUNDS",
        "human_review_required": False,
    }
    ranking = make_mock_ranking(
        actions_order=["HUMAN_REVIEW", "REMINDER", "PAYMENT_LINK", "RETRY", "NO_ACTION"],
        amount=500.0,
    )
    decision = policy_engine.evaluate_ranking(case, ranking)

    assert "HUMAN_REVIEW" in decision.blocked_actions
    assert "ERR_HUMAN_REVIEW_THRESHOLD_NOT_MET" in decision.blocked_actions["HUMAN_REVIEW"]
    assert decision.fallback_occurred is True
    assert decision.selected_action == "REMINDER"  # Rank 2 fallback


def test_high_amount_allows_human_review(policy_engine: PolicyEngine):
    """Verify high value case (>= INR 50k) allows HUMAN_REVIEW."""
    case = {
        "amount_at_risk": 75000.0,
        "retry_count": 0,
        "failure_reason": "PAYMENT_METHOD_DECLINED",
    }
    ranking = make_mock_ranking(
        actions_order=["HUMAN_REVIEW", "PAYMENT_LINK", "NO_ACTION", "REMINDER", "RETRY"],
        amount=75000.0,
    )
    decision = policy_engine.evaluate_ranking(case, ranking)

    assert decision.selected_action == "HUMAN_REVIEW"
    assert decision.is_approved is True
    assert decision.fallback_occurred is False


def test_ambiguous_failure_allows_human_review(policy_engine: PolicyEngine):
    """Verify AMBIGUOUS_FAILURE allows HUMAN_REVIEW."""
    case = {
        "amount_at_risk": 1200.0,
        "retry_count": 0,
        "failure_reason": "AMBIGUOUS_FAILURE",
    }
    ranking = make_mock_ranking(
        actions_order=["HUMAN_REVIEW", "NO_ACTION", "PAYMENT_LINK", "REMINDER", "RETRY"],
        amount=1200.0,
    )
    decision = policy_engine.evaluate_ranking(case, ranking)

    assert decision.selected_action == "HUMAN_REVIEW"
    assert decision.is_approved is True


def test_non_retryable_failure_blocks_retry(policy_engine: PolicyEngine):
    """Verify customer-action failures (e.g. CARD_EXPIRED) block RETRY."""
    case = {
        "amount_at_risk": 2000.0,
        "retry_count": 0,
        "failure_reason": "CARD_EXPIRED",
        "cooldown_eligible": True,
    }
    ranking = make_mock_ranking(
        actions_order=["RETRY", "PAYMENT_LINK", "REMINDER", "NO_ACTION", "HUMAN_REVIEW"],
        amount=2000.0,
    )
    decision = policy_engine.evaluate_ranking(case, ranking)

    assert "RETRY" in decision.blocked_actions
    assert "ERR_NON_RETRYABLE_FAILURE" in decision.blocked_actions["RETRY"]
    assert decision.selected_action == "PAYMENT_LINK"


def test_all_actions_blocked_falls_back_to_no_action(policy_engine: PolicyEngine):
    """Verify when all active actions are blocked, engine safely falls back to NO_ACTION."""
    case = {
        "amount_at_risk": 30000.0,  # Exceeds automatic limit
        "retry_count": 2,          # Exceeds retry limit
        "cooldown_eligible": False, # Cooldown active
        "failure_reason": "INSUFFICIENT_FUNDS",
        "human_review_required": False,
    }
    # Rank all active actions before NO_ACTION
    ranking = make_mock_ranking(
        actions_order=["HUMAN_REVIEW", "RETRY", "PAYMENT_LINK", "REMINDER", "NO_ACTION"],
        amount=30000.0,
    )
    decision = policy_engine.evaluate_ranking(case, ranking)

    assert decision.selected_action == "NO_ACTION"
    assert decision.fallback_occurred is True
