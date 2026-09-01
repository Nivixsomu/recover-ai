"""Action selection and expected-value ranking component for RecoverAI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml.src.features import (
    ALL_FEATURES,
    RECOVERY_ACTIONS,
    create_action_candidate_rows,
)


@dataclass(frozen=True)
class ActionCandidateScore:
    """Individual candidate action evaluation score."""

    action: str
    predicted_probability: float
    expected_recovery_value: float
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionRankingResult:
    """Comprehensive ranking result for a recovery opportunity."""

    amount_at_risk: float
    top_action: str
    top_probability: float
    top_expected_value: float
    rankings: List[ActionCandidateScore]
    case_metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "amount_at_risk": self.amount_at_risk,
            "top_action": self.top_action,
            "top_probability": self.top_probability,
            "top_expected_value": self.top_expected_value,
            "rankings": [r.to_dict() for r in self.rankings],
            "case_metadata": self.case_metadata,
        }


class ActionRanker:
    """Evaluates and ranks candidate recovery actions for payment opportunities."""

    def __init__(
        self,
        model: Any,
        actions: Sequence[str] = RECOVERY_ACTIONS,
    ) -> None:
        """Initialize ActionRanker with a fitted action-conditioned model.

        Args:
            model: Fitted sklearn Pipeline or CalibratedClassifierCV predicting
                   P(recovery_success | pre_intervention_features, action).
            actions: List of supported candidate actions.
        """
        self.model = model
        self.actions = tuple(actions)

    def rank_case(
        self,
        case_data: Union[pd.DataFrame, pd.Series, dict],
    ) -> ActionRankingResult:
        """Rank all 5 candidate actions for a single recovery opportunity.

        Evaluates:
            expected_recovery_value = P(success | X, action) * amount_at_risk

        Args:
            case_data: Pre-intervention features as a dict, Series, or 1-row DataFrame.

        Returns:
            ActionRankingResult with ranked candidate actions.
        """
        # Create exactly 5 candidate rows with identical context
        candidates_df = create_action_candidate_rows(case_data, actions=self.actions)
        amount_at_risk = float(candidates_df["amount_at_risk"].iloc[0])

        # Predict probability of recovery success (class 1)
        probabilities = self.model.predict_proba(candidates_df)[:, 1]

        scores: List[ActionCandidateScore] = []
        for action, prob in zip(self.actions, probabilities):
            prob_val = float(max(0.0, min(1.0, prob)))
            ev_val = float(prob_val * amount_at_risk)
            scores.append(
                ActionCandidateScore(
                    action=action,
                    predicted_probability=round(prob_val, 4),
                    expected_recovery_value=round(ev_val, 2),
                    rank=0,  # assigned after sorting
                )
            )

        # Sort by expected recovery value descending (tie-break by probability)
        scores.sort(
            key=lambda s: (s.expected_recovery_value, s.predicted_probability),
            reverse=True,
        )

        ranked_scores = [
            ActionCandidateScore(
                action=s.action,
                predicted_probability=s.predicted_probability,
                expected_recovery_value=s.expected_recovery_value,
                rank=idx + 1,
            )
            for idx, s in enumerate(scores)
        ]

        top = ranked_scores[0]
        meta = {
            k: v
            for k, v in (case_data if isinstance(case_data, dict) else case_data.to_dict()).items()
            if k in ("recovery_case_id", "payment_id", "failure_reason", "payment_method", "customer_segment")
        }

        return ActionRankingResult(
            amount_at_risk=amount_at_risk,
            top_action=top.action,
            top_probability=top.predicted_probability,
            top_expected_value=top.expected_recovery_value,
            rankings=ranked_scores,
            case_metadata=meta,
        )

    def rank_dataset(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Batch-evaluate candidate actions across all cases in a dataset.

        Args:
            df: DataFrame containing pre-intervention features for multiple cases.

        Returns:
            DataFrame containing optimal ML-recommended action, predicted probability,
            and expected recovery value per case.
        """
        results: List[Dict[str, Any]] = []

        # Vectorized batch evaluation across all 5 actions
        all_candidates_list: List[pd.DataFrame] = []
        for action in self.actions:
            action_df = df[ALL_FEATURES].copy()
            action_df["observed_action"] = action
            action_df["_candidate_action"] = action
            action_df["_case_index"] = np.arange(len(df))
            all_candidates_list.append(action_df)

        combined_candidates = pd.concat(all_candidates_list, ignore_index=True)
        probs = self.model.predict_proba(combined_candidates[ALL_FEATURES])[:, 1]
        combined_candidates["predicted_prob"] = probs
        combined_candidates["expected_value"] = combined_candidates["predicted_prob"] * combined_candidates["amount_at_risk"]

        # Pivot or groupby to find optimal action per case index
        optimal_indices = (
            combined_candidates.sort_values(by=["expected_value", "predicted_prob"], ascending=False)
            .groupby("_case_index")
            .head(1)
            .sort_values(by="_case_index")
        )

        output_df = df.copy()
        output_df["ml_selected_action"] = optimal_indices["_candidate_action"].values
        output_df["ml_predicted_probability"] = optimal_indices["predicted_prob"].values
        output_df["ml_expected_recovery_value"] = optimal_indices["expected_value"].values

        return output_df
