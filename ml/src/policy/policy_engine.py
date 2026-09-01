"""Enterprise Policy Engine for bounded, safe, and explainable recovery execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ml.src.data.schemas import (
    CUSTOMER_ACTION_FAILURES,
    FAILURE_REASONS,
    RECOVERY_ACTIONS,
    RETRYABLE_FAILURES,
    REVIEW_FAILURES,
)
from ml.src.models.action_ranker import ActionCandidateScore, ActionRankingResult


@dataclass(frozen=True)
class PolicyConfig:
    """Centralized, configurable business and safety thresholds."""

    max_automatic_retries: int = 2
    automatic_action_amount_limit: float = 25_000.0
    human_review_amount_threshold: float = 50_000.0
    human_review_retry_threshold: int = 3
    human_review_previous_failure_threshold: int = 20
    low_value_no_action_threshold: float = 300.0
    cooldown_hours: float = 24.0
    max_actions_per_case: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyEvaluationResult:
    """Immutable record of the policy evaluation and action filtering decision."""

    original_recommended_action: str
    selected_action: str
    is_approved: bool
    fallback_occurred: bool
    blocked_actions: Dict[str, str] = field(default_factory=dict)
    policy_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_recommended_action": self.original_recommended_action,
            "selected_action": self.selected_action,
            "is_approved": self.is_approved,
            "fallback_occurred": self.fallback_occurred,
            "blocked_actions": self.blocked_actions,
            "policy_reasons": self.policy_reasons,
        }


class PolicyEngine:
    """Evaluates ML-ranked candidate recovery actions against operational safety rules."""

    def __init__(self, config: Optional[PolicyConfig] = None) -> None:
        self.config = config or PolicyConfig()

    def evaluate_action_legality(
        self,
        action: str,
        case_data: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """Check if a specific action is legally and operationally permissible for a case.

        Args:
            action: Candidate action name (e.g. 'RETRY', 'HUMAN_REVIEW').
            case_data: Context dictionary for the recovery opportunity.

        Returns:
            Tuple of (is_allowed, block_reason_code_if_blocked)
        """
        amount = float(case_data.get("amount_at_risk", case_data.get("amount", 0.0)))
        retry_count = int(case_data.get("retry_count", 0))
        previous_failures = int(case_data.get("previous_failure_count", 0))
        failure_reason = str(case_data.get("failure_reason", ""))
        cooldown_eligible = bool(case_data.get("cooldown_eligible", True))
        human_review_required = bool(case_data.get("human_review_required", False))

        if action not in RECOVERY_ACTIONS:
            return False, "ERR_INVALID_ACTION"

        # Rule 1: NO_ACTION is always permissible as safe passive fallback
        if action == "NO_ACTION":
            return True, None

        # Rule 2: RETRY constraints
        if action == "RETRY":
            if retry_count >= self.config.max_automatic_retries:
                return False, f"ERR_RETRY_LIMIT_EXCEEDED (retry_count {retry_count} >= {self.config.max_automatic_retries})"
            if retry_count > 0 and not cooldown_eligible:
                return False, "ERR_COOLDOWN_NOT_MET (mandatory cooldown active)"
            if amount > self.config.automatic_action_amount_limit:
                return False, f"ERR_AMOUNT_EXCEEDS_AUTONOMY_LIMIT (INR {amount:,.2f} > {self.config.automatic_action_amount_limit:,.2f})"
            if failure_reason in CUSTOMER_ACTION_FAILURES:
                return False, f"ERR_NON_RETRYABLE_FAILURE ({failure_reason} requires customer action, not backend retry)"
            return True, None

        # Rule 3: HUMAN_REVIEW constraints
        if action == "HUMAN_REVIEW":
            # Allowed if explicit high-risk conditions are met
            meets_threshold = (
                human_review_required
                or amount >= self.config.human_review_amount_threshold
                or retry_count >= self.config.human_review_retry_threshold
                or previous_failures >= self.config.human_review_previous_failure_threshold
                or failure_reason in REVIEW_FAILURES
            )
            if not meets_threshold:
                return False, (
                    f"ERR_HUMAN_REVIEW_THRESHOLD_NOT_MET (ticket under INR {self.config.human_review_amount_threshold:,.2f} "
                    f"and does not meet escalation criteria)"
                )
            return True, None

        # Rule 4: PAYMENT_LINK constraints
        if action == "PAYMENT_LINK":
            if amount > self.config.automatic_action_amount_limit and not human_review_required:
                return False, f"ERR_AMOUNT_EXCEEDS_AUTONOMY_LIMIT (INR {amount:,.2f} > {self.config.automatic_action_amount_limit:,.2f})"
            return True, None

        # Rule 5: REMINDER constraints
        if action == "REMINDER":
            if retry_count > 0 and not cooldown_eligible:
                return False, "ERR_COOLDOWN_NOT_MET (reminder requires cooldown interval)"
            return True, None

        return True, None

    def evaluate_ranking(
        self,
        case_data: Dict[str, Any] | pd.Series,
        ranking_result: ActionRankingResult,
    ) -> PolicyEvaluationResult:
        """Filter ML ranked candidate actions against policy rules and select the top allowed action.

        Args:
            case_data: Feature context dictionary or Series.
            ranking_result: Ranked candidate scores from ActionRanker.

        Returns:
            PolicyEvaluationResult with final selected action, blocked actions, and reasons.
        """
        case_dict = case_data if isinstance(case_data, dict) else case_data.to_dict()
        top_ml_action = ranking_result.top_action

        blocked_actions: Dict[str, str] = {}
        policy_reasons: List[str] = []
        selected_action: Optional[str] = None
        fallback_occurred = False

        # Evaluate legality for all candidates to provide comprehensive policy inspection
        for candidate in ranking_result.rankings:
            action = candidate.action
            allowed, block_reason = self.evaluate_action_legality(action, case_dict)

            if allowed:
                if selected_action is None:
                    selected_action = action
            else:
                blocked_actions[action] = block_reason or "ERR_POLICY_BLOCKED"

        if selected_action is None:
            # Universal safety fallback
            selected_action = "NO_ACTION"
            fallback_occurred = True
            policy_reasons.append("All active recovery actions were blocked by policy. Selected NO_ACTION fallback.")
        elif selected_action != top_ml_action:
            fallback_occurred = True
            policy_reasons.append(
                f"Top ML action ({top_ml_action}) was blocked by policy. "
                f"Fell back to next allowed optimal action ({selected_action})."
            )
        else:
            policy_reasons.append(f"Top ML action ({top_ml_action}) passed all policy constraints.")

        is_approved = not fallback_occurred and (top_ml_action == selected_action)

        return PolicyEvaluationResult(
            original_recommended_action=top_ml_action,
            selected_action=selected_action,
            is_approved=is_approved,
            fallback_occurred=fallback_occurred,
            blocked_actions=blocked_actions,
            policy_reasons=policy_reasons,
        )
