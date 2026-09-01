"""Analyze ML datasets for feature engineering phase."""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = PROJECT_ROOT / "ml" / "data" / "final"

# Load all three splits
df_train = pd.read_csv(FINAL_DIR / "recovery_train.csv")
df_val = pd.read_csv(FINAL_DIR / "recovery_validation.csv")
df_test = pd.read_csv(FINAL_DIR / "recovery_test.csv")

print("=" * 80)
print("DATASET SHAPES")
print("=" * 80)
print(f"Train: {df_train.shape}")
print(f"Validation: {df_val.shape}")
print(f"Test: {df_test.shape}")

print("\n" + "=" * 80)
print("COLUMN CONSISTENCY")
print("=" * 80)
print(f"Train columns: {len(df_train.columns)}")
print(f"Train columns match Val: {set(df_train.columns) == set(df_val.columns)}")
print(f"Train columns match Test: {set(df_train.columns) == set(df_test.columns)}")

print("\n" + "=" * 80)
print("MISSING VALUES (TRAIN)")
print("=" * 80)
missing = df_train.isna().sum()
missing_nonzero = missing[missing > 0]
if len(missing_nonzero) > 0:
    print(missing_nonzero)
else:
    print("No missing values")

print("\n" + "=" * 80)
print("ACTION DISTRIBUTION (TRAIN)")
print("=" * 80)
print(df_train['observed_action'].value_counts().sort_index())

print("\n" + "=" * 80)
print("TARGET DISTRIBUTION (TRAIN)")
print("=" * 80)
print(df_train['recovery_success'].value_counts())
print(f"Recovery rate: {df_train['recovery_success'].mean():.2%}")

print("\n" + "=" * 80)
print("FAILURE REASON DISTRIBUTION (TRAIN)")
print("=" * 80)
print(df_train['failure_reason'].value_counts().sort_index())

print("\n" + "=" * 80)
print("CUSTOMER OVERLAP CHECK")
print("=" * 80)
train_cust = set(df_train['customer_id'])
val_cust = set(df_val['customer_id'])
test_cust = set(df_test['customer_id'])
print(f"Train customers: {len(train_cust)}")
print(f"Validation customers: {len(val_cust)}")
print(f"Test customers: {len(test_cust)}")
print(f"Train-Val overlap: {len(train_cust & val_cust)}")
print(f"Train-Test overlap: {len(train_cust & test_cust)}")
print(f"Val-Test overlap: {len(val_cust & test_cust)}")

print("\n" + "=" * 80)
print("CATEGORICAL COLUMNS UNIQUE VALUES")
print("=" * 80)
for col in df_train.select_dtypes(include=['object']).columns:
    n_unique = df_train[col].nunique()
    print(f"{col}: {n_unique} unique values")
    if n_unique <= 20:
        print(f"  Values: {sorted(df_train[col].unique())}")

print("\n" + "=" * 80)
print("NUMERICAL COLUMNS RANGES")
print("=" * 80)
for col in df_train.select_dtypes(include=['float64', 'int64']).columns:
    print(f"{col}: min={df_train[col].min():.2f}, max={df_train[col].max():.2f}")

print("\n" + "=" * 80)
print("ALL COLUMNS")
print("=" * 80)
for i, col in enumerate(df_train.columns, 1):
    print(f"{i:2d}. {col:40s} {str(df_train[col].dtype):10s}")
