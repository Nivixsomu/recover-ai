"""Core Recovery Service coordinating ML inference, PolicyEngine, explainability, audit, and execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

import joblib
import pandas as pd

from backend.app.integrations.razorpay import RazorpayActionExecutionResult, RazorpayTestClient
from backend.app.services.audit_service import AuditService
from ml.src.models import ActionRanker, ModelExplainer
from ml.src.policy import PolicyConfig, PolicyEngine, PolicyEvaluationResult

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "ml" / "models"


class RecoveryService:
    """End-to-end recovery service orchestrating the safe revenue recovery lifecycle."""

    _instance: Optional[RecoveryService] = None

    def __init__(self) -> None:
        model_path = MODELS_DIR / "final_recovery_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Champion model artifact not found at {model_path}")

        self.model = joblib.load(model_path)
        self.model_version = "Calibrated_HistGradientBoosting_Sigmoid_v2.0"
        self.ranker = ActionRanker(model=self.model)
        self.policy_engine = PolicyEngine(config=PolicyConfig())
        self.explainer = ModelExplainer(model_pipeline=self.model)
        self.audit_service = AuditService()
        self.razorpay_client = RazorpayTestClient()

    @classmethod
    def get_instance(cls) -> RecoveryService:
        """Singleton accessor for RecoveryService."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _normalize_case_input(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all required feature fields have valid non-None values."""
        case = payload.copy()
        if not case.get("recovery_case_id"):
            case["recovery_case_id"] = f"CASE-{uuid.uuid4().hex[:8].upper()}"

        amount_val = float(case.get("amount_at_risk") if case.get("amount_at_risk") is not None else case.get("amount", 1000.0))
        case["amount_at_risk"] = amount_val

        if not case.get("payment_id"):
            case["payment_id"] = f"pay_{uuid.uuid4().hex[:10]}"
        if not case.get("customer_id"):
            case["customer_id"] = "CUST-00001"
        if not case.get("customer_segment"):
            case["customer_segment"] = "NORMAL"
        if case.get("historical_success_rate") is None:
            case["historical_success_rate"] = 0.70
        if case.get("customer_transaction_frequency") is None:
            case["customer_transaction_frequency"] = 3.5
        if case.get("historical_transaction_count") is None:
            case["historical_transaction_count"] = 10
        if case.get("historical_average_amount") is None:
            case["historical_average_amount"] = amount_val
        if case.get("customer_lifetime_value") is None:
            case["customer_lifetime_value"] = amount_val * 10
        if case.get("previous_failure_count") is None:
            case["previous_failure_count"] = 0
        if case.get("previous_success_count") is None:
            case["previous_success_count"] = 7
        if case.get("retry_count") is None:
            case["retry_count"] = 0
        if case.get("attempt_count") is None:
            case["attempt_count"] = int(case.get("retry_count", 0)) + 1
        if case.get("time_since_previous_payment_hours") is None:
            case["time_since_previous_payment_hours"] = 24.0
        if case.get("hour_of_day") is None:
            case["hour_of_day"] = 14
        if case.get("day_of_week") is None:
            case["day_of_week"] = 2
        if case.get("cooldown_eligible") is None:
            case["cooldown_eligible"] = True
        if case.get("amount_limit_eligible") is None:
            case["amount_limit_eligible"] = bool(amount_val <= 25000.0)
        if case.get("human_review_required") is None:
            case["human_review_required"] = False

        return case

    def predict(self, case_input: Dict[str, Any]) -> Dict[str, Any]:
        """Run ML inference and generate raw action probabilities."""
        case_data = self._normalize_case_input(case_input)
        ranking_result = self.ranker.rank_case(case_data)

        probabilities = {
            r.action: r.predicted_probability for r in ranking_result.rankings
        }

        return {
            "recovery_case_id": case_data["recovery_case_id"],
            "model_version": self.model_version,
            "probabilities": probabilities,
            "top_action": ranking_result.top_action,
            "top_probability": ranking_result.top_probability,
            "expected_recovery_value": ranking_result.top_expected_value,
        }

    def recommend(self, case_input: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate ML candidate rankings, apply PolicyEngine constraints, and generate explainability (Dry Run)."""
        return self.process_and_execute(
            case_input=case_input,
            execute=False,
            idempotency_key=None,
        )

    def process_and_execute(
        self,
        case_input: Dict[str, Any],
        execute: bool = False,
        idempotency_key: Optional[str] = None,
        simulate_failure: bool = False,
    ) -> Dict[str, Any]:
        """Execute full safe recovery workflow: ML -> Ranker -> PolicyEngine -> Audit -> Razorpay Test Mode."""
        case_data = self._normalize_case_input(case_input)
        case_id = case_data["recovery_case_id"]

        # Step 1: Save / Record Case
        self.audit_service.save_case(case_data)
        self.audit_service.record_audit_event(case_id, "CASE_RECEIVED", case_data)

        # Step 2: ML Expected-Value Ranking
        ranking_result = self.ranker.rank_case(case_data)
        probabilities = {r.action: r.predicted_probability for r in ranking_result.rankings}
        rankings_list = [r.to_dict() for r in ranking_result.rankings]

        self.audit_service.save_prediction(
            recovery_case_id=case_id,
            model_version=self.model_version,
            probabilities=probabilities,
            rankings=rankings_list,
            top_action=ranking_result.top_action,
            top_probability=ranking_result.top_probability,
            top_expected_value=ranking_result.top_expected_value,
        )
        self.audit_service.record_audit_event(
            case_id,
            "ML_RANKING_EVALUATED",
            {
                "top_action": ranking_result.top_action,
                "top_probability": ranking_result.top_probability,
                "top_expected_value": ranking_result.top_expected_value,
                "rankings": rankings_list,
            },
        )

        # Step 3: PolicyEngine Evaluation
        policy_decision = self.policy_engine.evaluate_ranking(case_data, ranking_result)

        self.audit_service.save_policy_decision(
            recovery_case_id=case_id,
            original_action=policy_decision.original_recommended_action,
            selected_action=policy_decision.selected_action,
            is_approved=policy_decision.is_approved,
            fallback_occurred=policy_decision.fallback_occurred,
            blocked_actions=policy_decision.blocked_actions,
            policy_reasons=policy_decision.policy_reasons,
        )
        self.audit_service.record_audit_event(
            case_id,
            "POLICY_EVALUATED",
            policy_decision.to_dict(),
        )

        # Step 4: Decision Explanation
        explanation = self.explainer.explain_case_decision(ranking_result, case_data)
        if policy_decision.fallback_occurred:
            explanation["decision_reasons"].insert(
                0,
                f"Policy Engine rerouted decision from {policy_decision.original_recommended_action} to {policy_decision.selected_action} due to policy constraints."
            )

        # Step 5: Action Execution (Dry Run or Test Mode Dispatch)
        final_action = policy_decision.selected_action
        amount_at_risk = float(case_data["amount_at_risk"])

        if execute:
            exec_result = self.razorpay_client.execute_recovery_action(
                action=final_action,
                recovery_case_id=case_id,
                amount=amount_at_risk,
                idempotency_key=idempotency_key,
                dry_run=False,
                simulate_failure=simulate_failure,
            )
        else:
            exec_result = self.razorpay_client.execute_recovery_action(
                action=final_action,
                recovery_case_id=case_id,
                amount=amount_at_risk,
                idempotency_key=idempotency_key,
                dry_run=True,
            )

        self.audit_service.save_execution(
            recovery_case_id=case_id,
            action=exec_result.action,
            status=exec_result.status,
            reference_id=exec_result.reference_id,
            link_url=exec_result.link_url,
            message=exec_result.message,
            idempotency_key=idempotency_key,
        )
        self.audit_service.record_audit_event(
            case_id,
            "ACTION_EXECUTED",
            exec_result.model_dump(),
        )

        # Return full structured response
        return {
            "recovery_case_id": case_id,
            "amount_at_risk": amount_at_risk,
            "failure_reason": case_data.get("failure_reason"),
            "payment_method": case_data.get("payment_method"),
            "model_recommendation": {
                "top_action": ranking_result.top_action,
                "top_probability": ranking_result.top_probability,
                "top_expected_value": ranking_result.top_expected_value,
                "candidate_rankings": rankings_list,
            },
            "policy_decision": policy_decision.to_dict(),
            "selected_action": final_action,
            "explanation": explanation,
            "execution": exec_result.model_dump(),
            "audit_trail_url": f"/api/v1/recovery/{case_id}/audit",
        }
