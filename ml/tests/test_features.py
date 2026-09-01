"""Tests for feature engineering pipeline and preprocessor contract."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.src.features import (
    ALL_FEATURES,
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    EXCLUDED_COLUMNS,
    NUMERICAL_FEATURES,
    NUMERICAL_LOG_FEATURES,
    OTHER_NUMERICAL,
    RECOVERY_ACTIONS,
    TARGET_COLUMN,
    TEMPORAL_FEATURES,
    build_preprocessor,
    create_action_candidate_rows,
    fit_and_transform,
    get_feature_names_after_transform,
    load_and_prepare_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = PROJECT_ROOT / "ml" / "data" / "final"


@pytest.fixture(scope="module")
def datasets():
    """Load all three dataset splits."""
    return load_and_prepare_data(
        str(FINAL_DIR / "recovery_train.csv"),
        str(FINAL_DIR / "recovery_validation.csv"),
        str(FINAL_DIR / "recovery_test.csv"),
    )


def test_split_sizes(datasets):
    """Verify split sizes match specification."""
    X_train, y_train, X_val, y_val, X_test, y_test = datasets

    assert len(X_train) == 14000, f"Train size mismatch: {len(X_train)}"
    assert len(X_val) == 3000, f"Validation size mismatch: {len(X_val)}"
    assert len(X_test) == 3000, f"Test size mismatch: {len(X_test)}"

    assert len(y_train) == 14000
    assert len(y_val) == 3000
    assert len(y_test) == 3000


def test_all_features_present(datasets):
    """Verify all 20 model features are present and exact count is 20."""
    X_train, _, _, _, _, _ = datasets
    for col in ALL_FEATURES:
        assert col in X_train.columns, f"Feature {col} missing from dataset"
    assert len(ALL_FEATURES) == 20, f"Expected 20 features, found {len(ALL_FEATURES)}"


def test_feature_group_counts():
    """Verify feature group partitioning matches the specification."""
    assert len(NUMERICAL_FEATURES) == 6
    assert len(NUMERICAL_LOG_FEATURES) == 4
    assert len(CATEGORICAL_FEATURES) == 4
    assert len(TEMPORAL_FEATURES) == 2
    assert len(BOOLEAN_FEATURES) == 3
    assert len(OTHER_NUMERICAL) == 1
    assert len(ALL_FEATURES) == 20


def test_target_column_present_and_separated(datasets):
    """Verify target column exists in y and is strictly absent from X."""
    X_train, y_train, X_val, y_val, X_test, y_test = datasets

    for X in (X_train, X_val, X_test):
        assert TARGET_COLUMN not in X.columns, f"{TARGET_COLUMN} must not be in feature matrix X"

    for y in (y_train, y_val, y_test):
        assert isinstance(y, pd.Series)
        assert y.dtype in [int, np.int64]
        assert set(y.unique()) == {0, 1}, "Target must be binary 0/1"


def test_no_missing_values_in_features(datasets):
    """Verify no NaN values in pre-intervention features across all splits."""
    X_train, _, X_val, _, X_test, _ = datasets
    for name, X in [("train", X_train), ("validation", X_val), ("test", X_test)]:
        missing = X[ALL_FEATURES].isna().sum()
        assert missing.sum() == 0, f"Found {missing.sum()} missing values in {name} features"


def test_excluded_columns_not_used():
    """Verify excluded columns are strictly absent from feature set."""
    for col in EXCLUDED_COLUMNS:
        assert col not in ALL_FEATURES, f"Excluded column {col} found in feature set"


def test_action_column_in_features(datasets):
    """Verify action column is included as a feature with all 5 valid actions."""
    X_train, _, _, _, _, _ = datasets
    assert "observed_action" in ALL_FEATURES, "Action column must be a model feature"
    unique_actions = set(X_train["observed_action"].unique())
    assert unique_actions == set(RECOVERY_ACTIONS), f"Expected 5 actions, got {unique_actions}"


def test_leakage_columns_excluded():
    """Verify outcome and counterfactual columns are excluded from feature set."""
    leakage_cols = [
        "recovery_success",
        "recovered_amount",
        "time_to_recovery_hours",
        "potential_recovery_probability",
        "potential_recovery_success",
        "potential_recovered_amount",
        "potential_time_to_recovery_hours",
    ]
    for col in leakage_cols:
        assert col not in ALL_FEATURES, f"Leakage column {col} found in feature set"


def test_customer_id_and_identifiers_not_used():
    """Verify customer ID, payment ID, and case ID are not used as features."""
    for identifier in ["customer_id", "payment_id", "recovery_case_id"]:
        assert identifier not in ALL_FEATURES, f"Identifier {identifier} must not be used as feature"


def test_preprocessor_builds():
    """Verify preprocessor can be instantiated and has required methods."""
    preprocessor = build_preprocessor()
    assert preprocessor is not None
    assert hasattr(preprocessor, "fit_transform")
    assert hasattr(preprocessor, "transform")


def test_preprocessor_fit_transform(datasets):
    """Verify preprocessor can fit on train and transform all splits to 69 features."""
    X_train, _, X_val, _, X_test, _ = datasets
    preprocessor = build_preprocessor()

    # Fit strictly on training data
    X_train_transformed = preprocessor.fit_transform(X_train)
    assert isinstance(X_train_transformed, np.ndarray)
    assert X_train_transformed.shape == (14000, 69)

    # Transform validation and test
    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(X_test)

    assert X_val_transformed.shape == (3000, 69)
    assert X_test_transformed.shape == (3000, 69)


def test_no_nan_after_transform(datasets):
    """Verify transformed features have no NaN values."""
    X_train, _, X_val, _, X_test, _ = datasets
    preprocessor = build_preprocessor()

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(X_test)

    assert not np.isnan(X_train_transformed).any(), "NaN in train features"
    assert not np.isnan(X_val_transformed).any(), "NaN in validation features"
    assert not np.isnan(X_test_transformed).any(), "NaN in test features"


def test_transformed_shape_consistency(datasets):
    """Verify fit_and_transform helper produces matching dimensions."""
    X_train, y_train, X_val, y_val, X_test, y_test = datasets
    preprocessor = build_preprocessor()

    X_train_t, X_val_t, X_test_t = fit_and_transform(preprocessor, X_train, X_val, X_test)

    assert X_train_t.shape == (len(y_train), 69)
    assert X_val_t.shape == (len(y_val), 69)
    assert X_test_t.shape == (len(y_test), 69)


def test_categorical_consistency_across_splits(datasets):
    """Verify categorical handling is consistent across train/val/test."""
    X_train, _, X_val, _, X_test, _ = datasets

    for col in ["failure_reason", "payment_method", "customer_segment", "observed_action"]:
        train_cats = set(X_train[col].unique())
        val_cats = set(X_val[col].unique())
        test_cats = set(X_test[col].unique())

        assert val_cats.issubset(train_cats), f"New categories in validation for {col}"
        assert test_cats.issubset(train_cats), f"New categories in test for {col}"


def test_target_distribution_valid(datasets):
    """Verify target distribution is balanced across splits."""
    _, y_train, _, y_val, _, y_test = datasets

    for split_name, y_split in [("train", y_train), ("val", y_val), ("test", y_test)]:
        rate = y_split.mean()
        assert 0.45 < rate < 0.60, f"Target rate in {split_name} out of expected range: {rate:.2%}"


def test_feature_names_retrieval(datasets):
    """Verify feature names retrieval matches the 69 transformed dimensions."""
    X_train, _, _, _, _, _ = datasets
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    feature_names = get_feature_names_after_transform(preprocessor)
    assert isinstance(feature_names, list)
    assert len(feature_names) == 69, f"Expected 69 feature names, got {len(feature_names)}"
    assert all(isinstance(f, str) for f in feature_names)


def test_deterministic_transformation(datasets):
    """Verify transformation is strictly deterministic."""
    X_train, _, X_val, _, _, _ = datasets

    preprocessor1 = build_preprocessor()
    preprocessor2 = build_preprocessor()

    X_train_t1 = preprocessor1.fit_transform(X_train)
    X_val_t1 = preprocessor1.transform(X_val)

    X_train_t2 = preprocessor2.fit_transform(X_train)
    X_val_t2 = preprocessor2.transform(X_val)

    np.testing.assert_array_almost_equal(X_train_t1, X_train_t2)
    np.testing.assert_array_almost_equal(X_val_t1, X_val_t2)


def test_unknown_categorical_handling(datasets):
    """Verify unseen categorical values at inference time are handled gracefully."""
    X_train, _, _, _, _, _ = datasets
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    # Create dummy case with unseen categories
    sample = X_train.iloc[[0]].copy()
    sample["customer_segment"] = "UNSEEN_SEGMENT"
    sample["failure_reason"] = "UNKNOWN_FAILURE"
    sample["payment_method"] = "CRYPTO"

    transformed = preprocessor.transform(sample)
    assert transformed.shape == (1, 69)
    assert not np.isnan(transformed).any()


def test_create_action_candidate_rows(datasets):
    """Verify candidate action generator expands a case to 5 evaluation rows."""
    X_train, _, _, _, _, _ = datasets
    sample_case = X_train.iloc[0]

    candidates_df = create_action_candidate_rows(sample_case)
    assert len(candidates_df) == 5
    assert set(candidates_df["observed_action"]) == set(RECOVERY_ACTIONS)
    assert list(candidates_df.columns) == ALL_FEATURES

    # Ensure pre-intervention features are unchanged across candidate rows
    for col in ALL_FEATURES:
        if col != "observed_action":
            assert candidates_df[col].nunique() == 1

    # Ensure preprocessor can transform candidate rows
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    transformed_candidates = preprocessor.transform(candidates_df)
    assert transformed_candidates.shape == (5, 69)
    assert not np.isnan(transformed_candidates).any()


def test_dataframe_immutability(datasets):
    """Verify preprocessing does not mutate original DataFrames."""
    X_train, _, X_val, _, _, _ = datasets
    X_train_orig = X_train.copy()
    X_val_orig = X_val.copy()

    preprocessor = build_preprocessor()
    _ = preprocessor.fit_transform(X_train)
    _ = preprocessor.transform(X_val)

    pd.testing.assert_frame_equal(X_train, X_train_orig)
    pd.testing.assert_frame_equal(X_val, X_val_orig)
