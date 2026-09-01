"""Feature engineering pipeline for RecoverAI ML model.

Implements the deterministic, leakage-safe feature contract for predicting
P(recovery_success | pre_intervention_features, action)
across 5 candidate recovery actions:
- RETRY
- REMINDER
- PAYMENT_LINK
- HUMAN_REVIEW
- NO_ACTION
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Supported recovery actions
RECOVERY_ACTIONS: Tuple[str, ...] = (
    "RETRY",
    "REMINDER",
    "PAYMENT_LINK",
    "HUMAN_REVIEW",
    "NO_ACTION",
)

# Pre-intervention feature columns (observable at decision time t0)
# Numerical features transformed via StandardScaler
NUMERICAL_FEATURES: list[str] = [
    "historical_success_rate",
    "customer_transaction_frequency",
    "historical_transaction_count",
    "retry_count",
    "previous_failure_count",
    "previous_success_count",
]

# Long-tailed numerical features transformed via log1p + StandardScaler
NUMERICAL_LOG_FEATURES: list[str] = [
    "historical_average_amount",
    "customer_lifetime_value",
    "amount_at_risk",
    "time_since_previous_payment_hours",
]

# Categorical features transformed via OneHotEncoder
CATEGORICAL_FEATURES: list[str] = [
    "customer_segment",
    "failure_reason",
    "payment_method",
    "observed_action",
]

# Temporal features transformed via OneHotEncoder
TEMPORAL_FEATURES: list[str] = [
    "hour_of_day",
    "day_of_week",
]

# Boolean features converted to integer (0/1)
BOOLEAN_FEATURES: list[str] = [
    "cooldown_eligible",
    "amount_limit_eligible",
    "human_review_required",
]

# Passthrough numerical features
OTHER_NUMERICAL: list[str] = [
    "attempt_count",
]

# All 20 model input features (19 pre-intervention features + 1 action conditioning feature)
ALL_FEATURES: list[str] = (
    NUMERICAL_FEATURES
    + NUMERICAL_LOG_FEATURES
    + CATEGORICAL_FEATURES
    + TEMPORAL_FEATURES
    + BOOLEAN_FEATURES
    + OTHER_NUMERICAL
)

# Target variable (strictly separated from features)
TARGET_COLUMN: str = "recovery_success"

# Excluded columns and justification:
# - recovery_case_id, payment_id, customer_id: identifiers (prevent memorization, no generalization)
# - customer_created_at, created_at, last_history_event_at: raw timestamps (captured in derived features)
# - currency: constant ('INR')
# - amount: duplicate of amount_at_risk
# - historical_failure_count: duplicate alias of previous_failure_count
# - historical_success_count: duplicate alias of previous_success_count
# - payment_status: post-intervention state indicator / collinear with failure_reason
# - split: dataset partition metadata
# - action_allowed: post-hoc policy constraint flag
# - recovered_amount: post-intervention outcome variable (leakage)
# - time_to_recovery_hours: post-intervention outcome variable (leakage)
EXCLUDED_COLUMNS: list[str] = [
    "recovery_case_id",
    "payment_id",
    "customer_id",
    "customer_created_at",
    "created_at",
    "last_history_event_at",
    "currency",
    "amount",
    "historical_failure_count",
    "historical_success_count",
    "payment_status",
    "split",
    "action_allowed",
    "recovered_amount",
    "time_to_recovery_hours",
]


def log_transform(X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """Apply log1p transformation to long-tailed numeric features."""
    return np.log1p(np.asarray(X, dtype=float))


def boolean_to_int(X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """Convert boolean features to integer 0/1."""
    return np.asarray(X, dtype=int)


def build_preprocessor() -> ColumnTransformer:
    """Build sklearn ColumnTransformer preprocessing pipeline.

    Applies:
    - StandardScaler to normalized numerical features.
    - FunctionTransformer(log1p) + StandardScaler to long-tailed numerical features.
    - OneHotEncoder(handle_unknown='ignore') to categorical and temporal features.
    - FunctionTransformer(boolean_to_int) to boolean features.
    - Passthrough to attempt_count.

    Must be fitted on training data only.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            # Standard scaling for normalized numerical features (6 columns)
            (
                "numerical_std",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
            # Log transform + scale for long-tailed numerical features (4 columns)
            (
                "numerical_log",
                Pipeline([
                    ("log", FunctionTransformer(log_transform, validate=False)),
                    ("scale", StandardScaler()),
                ]),
                NUMERICAL_LOG_FEATURES,
            ),
            # One-hot encode categorical features (24 columns)
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop=None,
                ),
                CATEGORICAL_FEATURES,
            ),
            # One-hot encode temporal features (31 columns)
            (
                "temporal",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop=None,
                ),
                TEMPORAL_FEATURES,
            ),
            # Keep boolean features as-is, convert to integer 0/1 (3 columns)
            (
                "boolean",
                FunctionTransformer(
                    boolean_to_int,
                    validate=False,
                ),
                BOOLEAN_FEATURES,
            ),
            # Keep other numerical features as-is (1 column)
            (
                "other_numerical",
                "passthrough",
                OTHER_NUMERICAL,
            ),
        ],
        remainder="drop",
    )
    return preprocessor


def load_and_prepare_data(
    train_path: str,
    val_path: str,
    test_path: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load dataset partitions, extract features X and target y without mutating data.

    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        test_path: Path to test CSV

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # Extract feature copies and target series
    X_train = df_train[ALL_FEATURES].copy()
    y_train = df_train[TARGET_COLUMN].copy()

    X_val = df_val[ALL_FEATURES].copy()
    y_val = df_val[TARGET_COLUMN].copy()

    X_test = df_test[ALL_FEATURES].copy()
    y_test = df_test[TARGET_COLUMN].copy()

    return X_train, y_train, X_val, y_val, X_test, y_test


def fit_and_transform(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit preprocessor on training data and transform all splits.

    Args:
        preprocessor: sklearn ColumnTransformer
        X_train: Training features DataFrame
        X_val: Validation features DataFrame
        X_test: Test features DataFrame

    Returns:
        Tuple of (X_train_transformed, X_val_transformed, X_test_transformed)
    """
    # Fit strictly on training data
    X_train_transformed = preprocessor.fit_transform(X_train)

    # Transform validation and test using train-fitted preprocessor only
    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(X_test)

    return X_train_transformed, X_val_transformed, X_test_transformed


def get_feature_names_after_transform(preprocessor: ColumnTransformer) -> list[str]:
    """Get exact feature names and ordering after ColumnTransformer preprocessing.

    Args:
        preprocessor: A fitted ColumnTransformer instance.

    Returns:
        List of transformed feature names in column order.
    """
    feature_names: list[str] = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "numerical_std":
            feature_names.extend(list(columns))
        elif name == "numerical_log":
            feature_names.extend([f"{col}_log" for col in columns])
        elif name == "categorical":
            if hasattr(transformer, "get_feature_names_out"):
                feature_names.extend(list(transformer.get_feature_names_out(columns)))
            else:
                feature_names.extend(list(columns))
        elif name == "temporal":
            if hasattr(transformer, "get_feature_names_out"):
                feature_names.extend(list(transformer.get_feature_names_out(columns)))
            else:
                feature_names.extend(list(columns))
        elif name == "boolean":
            feature_names.extend(list(columns))
        elif name == "other_numerical":
            feature_names.extend(list(columns))

    return feature_names


def create_action_candidate_rows(
    case_data: Union[pd.DataFrame, pd.Series, dict],
    actions: Sequence[str] = RECOVERY_ACTIONS,
) -> pd.DataFrame:
    """Generate 5 candidate rows for a single recovery case, one per recovery action.

    For counterfactual inference (Phase 7+), this creates an evaluation matrix
    where pre-intervention context is fixed while `observed_action` varies across
    all supported recovery actions.

    Args:
        case_data: Single case as a dict, pd.Series, or 1-row DataFrame.
        actions: Sequence of recovery action names (defaults to all 5 RECOVERY_ACTIONS).

    Returns:
        DataFrame of len(actions) rows with identical pre-intervention features
        and varying `observed_action`, containing all columns in ALL_FEATURES.
    """
    if isinstance(case_data, dict):
        base_df = pd.DataFrame([case_data])
    elif isinstance(case_data, pd.Series):
        base_df = pd.DataFrame([case_data.to_dict()])
    elif isinstance(case_data, pd.DataFrame):
        if len(case_data) != 1:
            raise ValueError(f"Expected single case row, got {len(case_data)} rows.")
        base_df = case_data.copy()
    else:
        raise TypeError(f"Unsupported case_data type: {type(case_data)}")

    # Ensure all required features are present
    missing = [c for c in ALL_FEATURES if c not in base_df.columns and c != "observed_action"]
    if missing:
        raise ValueError(f"Missing required features in case_data: {missing}")

    candidates = []
    for action in actions:
        row = base_df.copy()
        row["observed_action"] = action
        candidates.append(row[ALL_FEATURES])

    return pd.concat(candidates, ignore_index=True)
