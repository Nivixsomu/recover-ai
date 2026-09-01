"""Model training, calibration, and evaluation utilities for RecoverAI."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from ml.src.features import build_preprocessor


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) with equal-width probability bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper if i < n_bins - 1 else y_prob <= bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    amount_at_risk: Optional[pd.Series | np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Calculate comprehensive predictive and expected-revenue metrics.

    Args:
        y_true: Ground truth binary outcomes (0 or 1).
        y_prob: Predicted probability of recovery success (class 1).
        amount_at_risk: Optional monetary amount at risk for revenue metrics.
        threshold: Classification decision threshold.

    Returns:
        Dictionary of metric names and numeric values.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_arr, y_prob)),
        "pr_auc": float(average_precision_score(y_true_arr, y_prob)),
        "brier_score": float(brier_score_loss(y_true_arr, y_prob)),
        "log_loss": float(log_loss(y_true_arr, y_prob)),
        "ece": compute_expected_calibration_error(y_true_arr, y_prob, n_bins=10),
        "mean_predicted_prob": float(np.mean(y_prob)),
        "observed_success_rate": float(np.mean(y_true_arr)),
    }

    if amount_at_risk is not None:
        amount_arr = np.asarray(amount_at_risk, dtype=float)
        metrics["total_amount_at_risk"] = float(np.sum(amount_arr))
        metrics["expected_recovered_revenue"] = float(np.sum(y_prob * amount_arr))
        metrics["observed_recovered_revenue"] = float(np.sum(y_true_arr * amount_arr))

    return metrics


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: Optional[ColumnTransformer] = None,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 20260401,
) -> Pipeline:
    """Train Logistic Regression baseline model in an end-to-end sklearn Pipeline.

    Args:
        X_train: Pre-intervention + action feature DataFrame.
        y_train: Target series (recovery_success).
        preprocessor: Optional ColumnTransformer; builds default if None.
        C: Inverse regularization strength.
        max_iter: Maximum solver iterations.
        random_state: Random seed for reproducibility.

    Returns:
        Fitted Pipeline containing preprocessor and logistic regression classifier.
    """
    if preprocessor is None:
        preprocessor = build_preprocessor()

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                C=C,
                max_iter=max_iter,
                random_state=random_state,
                solver="lbfgs",
            ),
        ),
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


def train_hist_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: Optional[ColumnTransformer] = None,
    max_iter: int = 150,
    learning_rate: float = 0.1,
    min_samples_leaf: int = 20,
    max_depth: Optional[int] = None,
    random_state: int = 20260401,
) -> Pipeline:
    """Train HistGradientBoostingClassifier in an end-to-end sklearn Pipeline.

    Args:
        X_train: Pre-intervention + action feature DataFrame.
        y_train: Target series (recovery_success).
        preprocessor: Optional ColumnTransformer; builds default if None.
        max_iter: Maximum boosting iterations.
        learning_rate: Boosting learning rate.
        min_samples_leaf: Minimum samples per leaf.
        max_depth: Maximum tree depth.
        random_state: Random seed for reproducibility.

    Returns:
        Fitted Pipeline containing preprocessor and gradient boosting classifier.
    """
    if preprocessor is None:
        preprocessor = build_preprocessor()

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            HistGradientBoostingClassifier(
                max_iter=max_iter,
                learning_rate=learning_rate,
                min_samples_leaf=min_samples_leaf,
                max_depth=max_depth,
                random_state=random_state,
            ),
        ),
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


def calibrate_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "sigmoid",
    cv: int = 5,
) -> CalibratedClassifierCV:
    """Fit a calibrated classifier using out-of-fold predictions on training data.

    Args:
        model: Base sklearn Pipeline.
        X_train: Training features.
        y_train: Training targets.
        method: 'sigmoid' (Platt scaling) or 'isotonic'.
        cv: Number of cross-validation folds.

    Returns:
        Fitted CalibratedClassifierCV model.
    """
    calibrator = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv=cv,
    )
    calibrator.fit(X_train, y_train)
    return calibrator
