"""Tests for model training, calibration, action-ranking, and explainability."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ml.src.features import (
    ALL_FEATURES,
    RECOVERY_ACTIONS,
    load_and_prepare_data,
)
from ml.src.models import (
    ActionRanker,
    ModelExplainer,
    evaluate_predictions,
    train_logistic_regression,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
FINAL_DIR = PROJECT_ROOT / "ml" / "data" / "final"


@pytest.fixture(scope="module")
def sample_dataset():
    """Load train/validation datasets for testing."""
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data(
        str(FINAL_DIR / "recovery_train.csv"),
        str(FINAL_DIR / "recovery_validation.csv"),
        str(FINAL_DIR / "recovery_test.csv"),
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


@pytest.fixture(scope="module")
def loaded_models():
    """Load persisted model artifacts from ml/models/."""
    lr_model = joblib.load(MODELS_DIR / "logistic_regression_baseline.joblib")
    hgb_model = joblib.load(MODELS_DIR / "hist_gradient_boosting.joblib")
    final_model = joblib.load(MODELS_DIR / "final_recovery_model.joblib")
    return {"lr": lr_model, "hgb": hgb_model, "final": final_model}


def test_saved_model_artifacts_exist():
    """Verify required model files and metadata exist in ml/models/."""
    expected_files = [
        "logistic_regression_baseline.joblib",
        "logistic_regression_metadata.json",
        "hist_gradient_boosting.joblib",
        "hist_gradient_boosting_metadata.json",
        "final_recovery_model.joblib",
        "final_model_metadata.json",
        "ml_evaluation_summary.json",
    ]
    for filename in expected_files:
        path = MODELS_DIR / filename
        assert path.exists(), f"Model artifact {filename} missing from ml/models/"


def test_model_predict_proba_bounds(loaded_models, sample_dataset):
    """Verify all models produce valid probabilities in [0, 1]."""
    _, _, X_val, _, _, _ = sample_dataset
    sample = X_val.iloc[:50]

    for name, model in loaded_models.items():
        probs = model.predict_proba(sample)
        assert probs.shape == (50, 2), f"{name} predict_proba shape mismatch"
        assert (probs >= 0.0).all() and (probs <= 1.0).all(), f"{name} probabilities outside [0, 1]"
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, err_msg=f"{name} probs do not sum to 1")


def test_evaluate_predictions_metrics():
    """Verify evaluation metric calculations are mathematically sound."""
    y_true = np.array([1, 0, 1, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.4, 0.2])
    amounts = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

    metrics = evaluate_predictions(y_true, y_prob, amount_at_risk=amounts)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["total_amount_at_risk"] == 1500.0
    assert metrics["expected_recovered_revenue"] == (0.9*100 + 0.1*200 + 0.8*300 + 0.4*400 + 0.2*500)


def test_action_ranker_contract(loaded_models, sample_dataset):
    """Verify ActionRanker evaluates exactly 5 actions and ranks by expected value."""
    _, _, X_val, _, _, _ = sample_dataset
    ranker = ActionRanker(model=loaded_models["final"])

    case = X_val.iloc[0]
    result = ranker.rank_case(case)

    # 1. Exactly 5 candidate actions
    assert len(result.rankings) == 5
    actions_ranked = [r.action for r in result.rankings]
    assert set(actions_ranked) == set(RECOVERY_ACTIONS)

    # 2. Probability in [0, 1] and non-negative EV
    for r in result.rankings:
        assert 0.0 <= r.predicted_probability <= 1.0
        assert r.expected_recovery_value >= 0.0
        # Check EV formula consistency
        expected_ev = round(r.predicted_probability * result.amount_at_risk, 2)
        assert abs(r.expected_recovery_value - expected_ev) <= 0.05

    # 3. Strictly descending order by expected recovery value
    ev_values = [r.expected_recovery_value for r in result.rankings]
    assert ev_values == sorted(ev_values, reverse=True)

    # 4. Top action matches rank 1
    assert result.top_action == result.rankings[0].action
    assert result.top_expected_value == result.rankings[0].expected_recovery_value


def test_action_ranker_context_isolation(loaded_models, sample_dataset):
    """Verify candidate evaluation preserves identical case context across actions."""
    _, _, X_val, _, _, _ = sample_dataset
    ranker = ActionRanker(model=loaded_models["final"])
    case = X_val.iloc[5]

    result = ranker.rank_case(case)
    assert result.amount_at_risk == float(case["amount_at_risk"])


def test_action_ranker_batch_dataset(loaded_models, sample_dataset):
    """Verify ActionRanker batch dataset evaluation."""
    _, _, X_val, _, _, _ = sample_dataset
    ranker = ActionRanker(model=loaded_models["final"])
    small_df = X_val.iloc[:20].copy()

    ranked_df = ranker.rank_dataset(small_df)
    assert len(ranked_df) == 20
    assert "ml_selected_action" in ranked_df.columns
    assert "ml_predicted_probability" in ranked_df.columns
    assert "ml_expected_recovery_value" in ranked_df.columns
    assert set(ranked_df["ml_selected_action"]).issubset(set(RECOVERY_ACTIONS))


def test_model_explainer_global(loaded_models):
    """Verify ModelExplainer extracts global feature importances."""
    explainer = ModelExplainer(model_pipeline=loaded_models["lr"])
    global_imp = explainer.get_global_feature_importance()

    assert isinstance(global_imp, pd.DataFrame)
    assert len(global_imp) == 69  # all 69 transformed features
    assert "feature" in global_imp.columns
    assert "importance" in global_imp.columns
    assert "direction" in global_imp.columns


def test_model_explainer_local(loaded_models, sample_dataset):
    """Verify ModelExplainer provides structured per-decision rationale."""
    _, _, X_val, _, _, _ = sample_dataset
    ranker = ActionRanker(model=loaded_models["final"])
    explainer = ModelExplainer(model_pipeline=loaded_models["final"])

    case = X_val.iloc[0]
    result = ranker.rank_case(case)
    explanation = explainer.explain_case_decision(result, case)

    assert explanation["selected_action"] == result.top_action
    assert explanation["predicted_probability"] == result.top_probability
    assert isinstance(explanation["decision_reasons"], list)
    assert len(explanation["decision_reasons"]) >= 2
    assert "disclaimer" in explanation
