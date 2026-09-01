"""Model explainability and decision rationale for RecoverAI."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from ml.src.features import get_feature_names_after_transform
from ml.src.models.action_ranker import ActionRankingResult


class ModelExplainer:
    """Provides global feature importance and instance-level decision explanations."""

    def __init__(self, model_pipeline: Any) -> None:
        """Initialize with an end-to-end trained model pipeline."""
        self.pipeline = model_pipeline
        self.preprocessor = None
        self.classifier = None
        self._extract_pipeline_components()

    def _extract_pipeline_components(self) -> None:
        """Extract preprocessor and estimator from pipeline or calibrated wrapper."""
        current = self.pipeline
        # Handle CalibratedClassifierCV
        if hasattr(current, "estimator"):
            current = current.estimator
        if isinstance(current, Pipeline):
            self.preprocessor = current.named_steps.get("preprocessor")
            self.classifier = current.named_steps.get("classifier")
        elif hasattr(current, "named_steps"):
            self.preprocessor = current.named_steps.get("preprocessor")
            self.classifier = current.named_steps.get("classifier")

    def get_global_feature_importance(
        self,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        n_repeats: int = 5,
        random_state: int = 20260401,
    ) -> pd.DataFrame:
        """Compute global feature importances or coefficients.

        Returns:
            DataFrame with columns ['feature', 'importance', 'direction']
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not found in pipeline.")

        feature_names = get_feature_names_after_transform(self.preprocessor)

        if hasattr(self.classifier, "coef_"):
            # Linear model coefficients
            coefs = self.classifier.coef_[0]
            df_importance = pd.DataFrame({
                "feature": feature_names,
                "importance": np.abs(coefs),
                "raw_coefficient": coefs,
                "direction": ["POSITIVE" if c > 0 else "NEGATIVE" for c in coefs],
            }).sort_values(by="importance", ascending=False).reset_index(drop=True)
            return df_importance

        if X_val is not None and y_val is not None:
            # Permutation importance for tree-based models on validation set
            perm = permutation_importance(
                self.pipeline,
                X_val,
                y_val,
                n_repeats=n_repeats,
                random_state=random_state,
                scoring="roc_auc",
            )
            df_importance = pd.DataFrame({
                "feature": list(X_val.columns),
                "importance": perm.importances_mean,
                "std": perm.importances_std,
            }).sort_values(by="importance", ascending=False).reset_index(drop=True)
            return df_importance

        return pd.DataFrame()

    def explain_case_decision(
        self,
        ranking_result: ActionRankingResult,
        case_data: pd.Series | dict,
    ) -> Dict[str, Any]:
        """Generate human-readable and structured per-decision explanation.

        Explains why the top action was selected over alternatives based on model
        predictions and business logic.
        """
        case_dict = case_data if isinstance(case_data, dict) else case_data.to_dict()
        top = ranking_result.rankings[0]
        runner_up = ranking_result.rankings[1] if len(ranking_result.rankings) > 1 else None

        reasons = [
            f"Predicted recovery probability for {top.action} is {top.predicted_probability:.1%}.",
            f"Expected recovered revenue under {top.action} is INR {top.expected_recovery_value:,.2f}.",
        ]

        if runner_up:
            prob_diff = top.predicted_probability - runner_up.predicted_probability
            ev_diff = top.expected_recovery_value - runner_up.expected_recovery_value
            reasons.append(
                f"Outranked next-best action ({runner_up.action}) by +INR {ev_diff:,.2f} "
                f"expected value (+{prob_diff:+.1%} probability difference)."
            )

        # Context drivers
        context_factors = {
            "failure_reason": case_dict.get("failure_reason"),
            "payment_method": case_dict.get("payment_method"),
            "customer_segment": case_dict.get("customer_segment"),
            "retry_count": case_dict.get("retry_count"),
            "amount_at_risk": case_dict.get("amount_at_risk"),
            "historical_success_rate": case_dict.get("historical_success_rate"),
        }

        return {
            "selected_action": top.action,
            "predicted_probability": top.predicted_probability,
            "expected_recovery_value": top.expected_recovery_value,
            "runner_up_action": runner_up.action if runner_up else None,
            "runner_up_expected_value": runner_up.expected_recovery_value if runner_up else None,
            "decision_reasons": reasons,
            "context_factors": context_factors,
            "disclaimer": (
                "Explanation reflects synthetic simulator probability relationships "
                "and does not represent production real-world payment causality."
            ),
        }
