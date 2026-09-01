# RecoverAI ML Feature Specification

**Phase:** 6B — ML Feature Engineering  
**Objective:** Action-conditioned recovery success prediction  
**Target Variable:** `recovery_success` (binary, 0/1)  
**Model Formulation:** `P(recovery_success | pre_intervention_features, action)`  
**Dataset:** recovery-simulator-v2 (seed: 20260401)  
**Train/Val/Test:** 14,000 / 3,000 / 3,000 cases  

---

## 1. ML OBJECTIVE

Build a supervised model that predicts recovery success conditioned on:

1. **Pre-intervention features** — observable at decision time t₀
2. **Action** — the recovery action taken (RETRY, REMINDER, PAYMENT_LINK, HUMAN_REVIEW, NO_ACTION)

The action must be included as a model feature, not removed or predicted separately.

**Later use (Phase 7+):** For a new recovery case:
1. Create 5 feature rows (one per candidate action)
2. Predict P(success | features, action) for each
3. Calculate expected value = P(success) × amount_at_risk
4. Rank actions and select the highest-value one
5. Send selected action to policy engine

---

## 2. DATASET COMPOSITION

| Split | Cases | Customers | Rows | Date Range |
|-------|-------|-----------|------|------------|
| Train | 14,000 | 2,800 | 14,000 | 2025-01-01 to 2025-09-30 |
| Validation | 3,000 | 600 | 3,000 | 2025-10-01 to 2025-11-15 |
| Test | 3,000 | 600 | 3,000 | 2025-11-16 to 2025-12-31 |
| **Total** | **20,000** | **4,000** | **20,000** | — |

**Customer Isolation:** ✓ VERIFIED
- Zero overlap between train/validation/test
- Each customer appears in only one split
- Prevents temporal leakage

**Columns:** 36 total (after train/val/test loading)
**Pre-intervention Features:** 19
**Action Column:** 1
**Target:** 1
**Excluded:** 15

---

## 3. EXACT DATASET SCHEMA

### A. TARGET VARIABLE

| Column | Type | Values | Distribution (Train) | Notes |
|--------|------|--------|----------------------|-------|
| `recovery_success` | int64 (0/1) | 0, 1 | 0: 6,492 (46.37%), 1: 7,508 (53.63%) | Binary classification target. 0 = recovery failed. 1 = recovery succeeded. |

**Action Within Target:**

| Observed Action | Success Rate | Cases |
|-----------------|--------------|-------|
| HUMAN_REVIEW | 77.47% | 593 |
| REMINDER | 62.86% | 1,312 |
| PAYMENT_LINK | 54.95% | 5,232 |
| RETRY | 60.46% | 5,133 |
| NO_ACTION | 6.97% | 1,730 |

---

### B. ACTION FEATURE (CRITICAL)

| Column | Type | Values | Cardinality | Notes |
|--------|------|--------|-------------|-------|
| `observed_action` | str (categorical) | HUMAN_REVIEW, NO_ACTION, PAYMENT_LINK, REMINDER, RETRY | 5 | The recovery action taken. Must be included as a model feature. For inference, will create 5 copies of each case with each action, predict separately, then rank by expected value. |

---

### C. PRE-INTERVENTION FEATURES (19 total)

#### C1. CUSTOMER CONTEXT (6 features)

| Column | Type | Range | Missing | Feature Type | Purpose |
|--------|------|-------|---------|--------------|---------|
| `customer_segment` | str | HIGH_VALUE, LOW_FREQUENCY, NORMAL, OCCASIONAL_FAILURE, RELIABLE | 0 | Categorical | Customer behavioral segmentation from history. 5 categories. |
| `historical_success_rate` | float64 | [0.08, 1.00] | 0 | Numerical | Fraction of past transactions that succeeded. Key predictor of recoverability. |
| `historical_average_amount` | float64 | [113.18, 16044.54] | 0 | Numerical | Mean transaction value in customer history. Indicates customer spending level. |
| `customer_lifetime_value` | float64 | [345.45, 823351.98] | 0 | Numerical | Total historical revenue from customer. Long-tailed distribution. |
| `customer_transaction_frequency` | float64 | [0.06, 25.78] | 0 | Numerical | Transactions per month (derived from customer_created_at and history). Engagement signal. |
| `historical_transaction_count` | int64 | [2, 128] | 0 | Numerical | Total number of past transactions. History depth indicator. |

#### C2. PAYMENT FAILURE CONTEXT (8 features)

| Column | Type | Range | Missing | Feature Type | Purpose |
|--------|------|-------|---------|--------------|---------|
| `failure_reason` | str | 10 types: AMBIGUOUS_FAILURE, BANK_TIMEOUT, CARD_EXPIRED, CHECKOUT_ABANDONED, GATEWAY_TIMEOUT, INSUFFICIENT_FUNDS, INVALID_PAYMENT_DETAILS, NETWORK_ERROR, PAYMENT_METHOD_DECLINED, TEMPORARY_BANK_ERROR | 0 | Categorical | Root cause of payment failure. Strongly influences recoverability by action. 10 categories. |
| `payment_method` | str | CARD, NETBANKING, UPI, WALLET | 0 | Categorical | Payment instrument used. Affects retry/link success rates. 4 categories. |
| `amount_at_risk` | float64 | [100.00, 48509.03] | 0 | Numerical | Amount of failed payment (INR). Affects policy eligibility and recovery probability. Same as `amount`. |
| `retry_count` | int64 | [0, 3] | 0 | Numerical | Number of previous retry attempts. Diminishing returns. 0-3 range. |
| `previous_failure_count` | int64 | [0, 46] | 0 | Numerical | Number of prior failures in customer's history. Signals repeated payment problems. |
| `previous_success_count` | int64 | [1, 118] | 0 | Numerical | Number of successful prior payments. Contrasts with failures. |
| `historical_failure_count` | int64 | [0, 46] | 0 | Numerical | Derived: Total failures in history (same as previous_failure_count at feature time). |
| `historical_success_count` | int64 | [1, 118] | 0 | Numerical | Derived: Total successes in history (same as previous_success_count at feature time). |

#### C3. TEMPORAL & INTERACTION FEATURES (5 features)

| Column | Type | Range | Missing | Feature Type | Purpose |
|--------|------|-------|---------|--------------|---------|
| `time_since_previous_payment_hours` | float64 | [1.00, 30770.04] | 0 | Numerical | Hours since customer's last payment attempt. Long gaps may indicate account inactivity; short gaps indicate urgency. |
| `hour_of_day` | int64 | [0, 23] | 0 | Categorical/Numerical | Hour when payment failed (0-23). May affect recovery by action (e.g., reminder effectiveness). Treat as categorical (24 categories). |
| `day_of_week` | int64 | [0, 6] | 0 | Categorical/Numerical | Day of week when payment failed (0=Monday, 6=Sunday). May affect recovery timing. Treat as categorical (7 categories). |
| `attempt_count` | int64 | [1, 4] | 0 | Numerical | Total attempts including this failed one (retry_count + 1). Measure of persistence. |
| `cooldown_eligible` | bool | True, False | 0 | Boolean | Whether enough time has passed since previous attempt to retry. Policy eligibility signal. |

#### C4. POLICY ELIGIBILITY SIGNALS (2 features)

| Column | Type | Range | Missing | Feature Type | Purpose |
|--------|------|-------|---------|--------------|---------|
| `amount_limit_eligible` | bool | True, False | 0 | Boolean | Whether amount is ≤ INR 25,000 (autonomous action threshold). Policy constraint. |
| `human_review_required` | bool | True, False | 0 | Boolean | Whether policy mandates HUMAN_REVIEW (high amount, ambiguous failure, repeated issues). Policy constraint. |

---

### D. EXCLUDED COLUMNS & REASONS

| Column | Type | Reason for Exclusion | Risk Level |
|--------|------|----------------------|-----------|
| `recovery_case_id` | str | Unique identifier, no predictive value | **LEAKAGE RISK** (if used as feature) |
| `payment_id` | str | Unique identifier, no predictive value | **LEAKAGE RISK** (if used as feature) |
| `customer_id` | str | Unique identifier. Customer features already captured. Prevents generalization. | **LEAKAGE RISK** (if used as feature) |
| `customer_created_at` | str | Timestamp. Temporal information already captured via derived features. Cannot be used directly. | INDIRECT LEAKAGE |
| `created_at` | str | Timestamp. Information about decision time, already captured via hour_of_day, day_of_week. | INDIRECT LEAKAGE |
| `last_history_event_at` | str | Timestamp. Information already captured in time_since_previous_payment_hours. | INDIRECT LEAKAGE |
| `currency` | str | Constant value (INR). No variance. Not informative. | None (redundant) |
| `amount` | float64 | **DUPLICATE**: identical to `amount_at_risk`. Use only amount_at_risk. | None (redundancy) |
| `payment_status` | str | **POST-INTERVENTION INDICATOR**: derived from failure reason and payment state. Collinear with failure_reason. Not independent. | **LEAKAGE RISK** |
| `split` | str | Metadata field (train/validation/test). Not a predictive feature. | None (metadata) |
| `recovered_amount` | float64 | **OUTCOME VARIABLE**: only known after recovery attempt. Uses future information. | **CRITICAL LEAKAGE** |
| `time_to_recovery_hours` | float64 | **OUTCOME VARIABLE**: only known after recovery attempt. Uses future information. Missing for all failed cases. | **CRITICAL LEAKAGE** |
| `action_allowed` | bool | Derived post-hoc from policy constraints. Collinear with amount_limit_eligible and human_review_required. | **INDIRECT LEAKAGE** |

---

## 4. LEAKAGE AUDIT

### A. Confirmed Leakage-Free Features

✓ **All 19 pre-intervention features** are observable at decision time t₀, before any recovery action is taken.

✓ **`observed_action`** is part of the conditioning set, not derived from outcomes.

✓ No feature depends on recovery_success, recovered_amount, or time_to_recovery_hours.

✓ No future payment information included.

✓ No post-intervention fields included.

✓ No optimal-action derivation (which would depend on potential outcomes).

### B. Indirect Leakage Prevention

- **Timestamps:** Converted to derived features (hour_of_day, day_of_week, time_since_previous_payment_hours) rather than used raw. ✓
- **Customer IDs:** Excluded to prevent memorization. Customer-level information captured in historical aggregates. ✓
- **Payment Status:** Collinear with failure_reason; excluded. ✓

### C. Leakage Test During Training

When training, ensure:
1. No prediction features are fit on validation/test data.
2. Preprocessing (scaling, encoding) is fitted **only on training data**.
3. Validation/test data are transformed using train-fitted transformers.

---

## 5. PRE-INTERVENTION FEATURES SUMMARY

| Category | Count | Columns |
|----------|-------|---------|
| Customer Context | 6 | customer_segment, historical_success_rate, historical_average_amount, customer_lifetime_value, customer_transaction_frequency, historical_transaction_count |
| Failure Context | 5 | failure_reason, payment_method, amount_at_risk, previous_failure_count, previous_success_count |
| Temporal & Retry Context | 5 | retry_count, attempt_count, time_since_previous_payment_hours, hour_of_day, day_of_week |
| Policy Signals | 3 | cooldown_eligible, amount_limit_eligible, human_review_required |
| **ACTION (conditioning variable)** | **1** | **observed_action** |
| **TOTAL MODEL INPUTS** | **20** | (19 features + 1 action) |

---

## 6. FEATURE ENGINEERING PLAN

### A. Numerical Features (10 continuous + 1 discrete)

| Source Column | Transformation | Rationale | Fit on Train? |
|----------------|-----------------|-----------|---------------|
| `historical_success_rate` | StandardScaler | Core recoverability signal [0.08, 1.00]. | Yes (fit on train) |
| `customer_transaction_frequency` | StandardScaler | Monthly transaction rate [0.06, 25.78]. | Yes (fit on train) |
| `historical_transaction_count` | StandardScaler | History depth [2, 128]. | Yes (fit on train) |
| `retry_count` | StandardScaler | Retry count [0, 3]. | Yes (fit on train) |
| `previous_failure_count` | StandardScaler | Prior failure volume [0, 46]. | Yes (fit on train) |
| `previous_success_count` | StandardScaler | Prior success volume [1, 118]. | Yes (fit on train) |
| `historical_average_amount` | log1p + StandardScaler | Long-tailed customer spend [113.18, 16044.54]. | Yes (fit on train) |
| `customer_lifetime_value` | log1p + StandardScaler | Heavy long-tail [345.45, 823351.98]. | Yes (fit on train) |
| `amount_at_risk` | log1p + StandardScaler | Long-tailed amount [100.00, 48509.03]. | Yes (fit on train) |
| `time_since_previous_payment_hours` | log1p + StandardScaler | Wide dynamic range [1.00, 30770.04]. | Yes (fit on train) |
| `attempt_count` | Passthrough | Attempt count [1, 4]. Preserved as integer. | No |

**Scaling Strategy:**
- Use `sklearn.preprocessing.StandardScaler` for standard numerical features.
- Use `numpy.log1p` + `StandardScaler` for extreme long-tails (amounts, CLV, time gap).
- Fit scalers **strictly on training data** (14,000 cases).
- Transform validation (3,000) and test (3,000) with train-fitted scalers.

### B. Categorical Features (4 standard + 2 temporal)

| Source Column | Cardinality | Encoding | Fit on Train? | Notes |
|----------------|-------------|----------|---------------|-------|
| `customer_segment` | 5 | One-hot encoding | Yes | 5 binary features (HIGH_VALUE, LOW_FREQUENCY, NORMAL, OCCASIONAL_FAILURE, RELIABLE). |
| `failure_reason` | 10 | One-hot encoding | Yes | 10 binary features across all failure modes. |
| `payment_method` | 4 | One-hot encoding | Yes | 4 binary features (CARD, NETBANKING, UPI, WALLET). |
| `observed_action` | 5 | One-hot encoding | Yes | 5 binary features (HUMAN_REVIEW, NO_ACTION, PAYMENT_LINK, REMINDER, RETRY). |
| `hour_of_day` | 24 | One-hot encoding | Yes | 24 binary features for payment failure hour (0–23). |
| `day_of_week` | 7 | One-hot encoding | Yes | 7 binary features for failure day of week (0=Mon, 6=Sun). |

**Handling Unknown Categories:**
- Use `handle_unknown='ignore'` in `OneHotEncoder` to safely encode unseen production values as all zeros.

### C. Boolean Features (3 total)

| Source Column | Encoding | Notes |
|----------------|----------|-------|
| `cooldown_eligible` | Convert to integer (0/1) | Binary policy eligibility indicator. |
| `amount_limit_eligible` | Convert to integer (0/1) | Binary INR 25k autonomy constraint. |
| `human_review_required` | Convert to integer (0/1) | Binary policy mandate flag. |

---

## 7. FEATURE MATRIX STRUCTURE (After Engineering)

### Input Features (X)

| Category | Source Inputs | Transformed Columns | Details |
|----------|---|---|---------|
| Numerical (StandardScaler) | 6 | 6 | `historical_success_rate`, `customer_transaction_frequency`, `historical_transaction_count`, `retry_count`, `previous_failure_count`, `previous_success_count` |
| Numerical (log1p + StandardScaler) | 4 | 4 | `historical_average_amount`, `customer_lifetime_value`, `amount_at_risk`, `time_since_previous_payment_hours` |
| One-hot Categorical | 4 | 24 | `customer_segment` (5) + `failure_reason` (10) + `payment_method` (4) + `observed_action` (5) |
| One-hot Temporal | 2 | 31 | `hour_of_day` (24) + `day_of_week` (7) |
| Boolean (as int 0/1) | 3 | 3 | `cooldown_eligible`, `amount_limit_eligible`, `human_review_required` |
| Passthrough Numerical | 1 | 1 | `attempt_count` |
| **TOTALS** | **20 inputs** | **69 features** | Dense numeric feature matrix |

### Target Variable (y)

| Column | Type | Values | Distribution |
|--------|------|--------|---------------|
| `recovery_success` | Binary (0/1) | 0, 1 | Train: 46.37% / 53.63% (class balance: 0.868) |

**Class Imbalance:** Moderate (~1:1.15 ratio). May not require resampling, but can use class_weight='balanced' in model training.

---

## 8. PREPROCESSING PIPELINE (Conceptual)

```
Input CSV → Load Data
         ↓
Select Pre-Intervention Features + Action
         ↓
Split: Train / Validation / Test (do NOT resplit)
         ↓
FIT on Training Data Only:
  - StandardScaler for long-tailed numericals
  - OneHotEncoder for categoricals
  - Log transformer for amounts
         ↓
TRANSFORM All Splits:
  - Apply train-fitted transformers to validation & test
         ↓
Combine: Scaled + One-Hot → Feature Matrix X
         ↓
Target: Select recovery_success → y
         ↓
Model Input: (X_train, y_train), (X_val, y_val), (X_test, y_test)
```

**Code Structure (sklearn):**
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
    ]
)

# Fit ONLY on training data
X_train_transformed = preprocessor.fit_transform(X_train)
# Transform validation/test with train-fitted transformer
X_val_transformed = preprocessor.transform(X_val)
X_test_transformed = preprocessor.transform(X_test)
```

---

## 9. TRAIN / VALIDATION / TEST STRATEGY

### A. Data Splits (No Reshuffling)

| Split | Cases | Date Range | Purpose |
|-------|-------|-----------|---------|
| Train | 14,000 | 2025-01-01 to 2025-09-30 | Fit preprocessor and model |
| Validation | 3,000 | 2025-10-01 to 2025-11-15 | Hyperparameter tuning, early stopping |
| Test | 3,000 | 2025-11-16 to 2025-12-31 | Final evaluation, no tuning |

### B. Preprocessing Fitting Rules

| Component | Where to Fit | Where to Apply |
|-----------|--------------|-----------------|
| StandardScaler | Train only | Train + Validation + Test |
| OneHotEncoder | Train only | Train + Validation + Test |
| Log transformers | Train only (compute log1p bounds) | Train + Validation + Test |

### C. Model Training

- **Fit model:** X_train_transformed, y_train
- **Validate:** X_val_transformed, y_val (for early stopping, hyperparameter selection)
- **Test:** X_test_transformed, y_test (final held-out evaluation, no tuning)

### D. Customer Isolation

✓ VERIFIED: No customer appears in multiple splits.
- Train: 2,800 customers
- Validation: 600 customers (different from train)
- Test: 600 customers (different from train and validation)
- Zero overlap confirmed

---

## 10. COUNTERFACTUAL INFERENCE FLOW (Phase 7+)

### For a New Recovery Case (at t₀):

**Input:** One observed payment failure with pre-intervention features

**Process:**

```
1. Extract pre-intervention features for this case.
   X_case = {customer_segment, amount_at_risk, ..., failure_reason}

2. Create 5 candidate rows:
   candidates = [
       {X_case + action='RETRY'},
       {X_case + action='REMINDER'},
       {X_case + action='PAYMENT_LINK'},
       {X_case + action='HUMAN_REVIEW'},
       {X_case + action='NO_ACTION'},
   ]

3. Preprocess each candidate using train-fitted preprocessor:
   X_candidates_transformed = preprocessor.transform(candidates)

4. Predict P(success) for each:
   p_success = model.predict_proba(X_candidates_transformed)[:, 1]

5. Calculate expected recovery value:
   expected_value = p_success * case.amount_at_risk

6. Rank actions by expected value:
   best_action = candidates[argmax(expected_value)]

7. Send best_action to policy engine (Phase 7+).
```

**Key:** The model is **action-conditioned**, so it naturally supports this counterfactual evaluation.

---

## 11. DATA VALIDATION CHECKLIST

Before training:

- [ ] Train/Validation/Test shapes: (14000, X), (3000, X), (3000, X)
- [ ] All 36 columns present and consistent across splits
- [ ] No customer IDs in train/val/test overlap
- [ ] Target `recovery_success`: binary, no missing values
- [ ] All action values present: HUMAN_REVIEW, NO_ACTION, PAYMENT_LINK, REMINDER, RETRY
- [ ] All failure reasons present (10 types)
- [ ] All payment methods present (4 types)
- [ ] All customer segments present (5 types)
- [ ] No leakage columns (recovered_amount, time_to_recovery_hours, etc.) included in features
- [ ] No NaN values in pre-intervention feature columns (time_to_recovery_hours may be NaN, but it's excluded)
- [ ] Action distribution reasonable in each split
- [ ] Target distribution not severely imbalanced

---

## 12. LIMITATIONS

1. **Synthetic data:** Model will not generalize to real Razorpay production payment flows without retraining on real data.
2. **No intervention costs:** Recovered amount is simulated; actual intervention cost (SMS, payment link hosting, agent time) not modeled. Net revenue requires cost subtraction.
3. **Action conditionality:** Model assumes action is deterministic or externally imposed. Real systems may have feedback loops or adaptive policies.
4. **No cold-start for new customers:** Feature set requires historical transaction data. Customers with no history cannot be scored.
5. **Feature scale:** Features are normalized to train distribution. Inference data far outside train range may have degraded performance.
6. **Temporal patterns:** Train/val/test are chronological, not random. Time-series effects may influence results.

---

## 13. NEXT STEPS (PHASE 7+, DO NOT START NOW)

1. Implement feature preprocessing pipeline using this spec
2. Load train/validation/test data
3. Fit preprocessor on training data only
4. Select a supervised learning algorithm (e.g., logistic regression, gradient boosting)
5. Train model on transformed training data
6. Tune hyperparameters using validation set
7. Evaluate on held-out test set
8. Implement counterfactual inference (action ranking) pipeline
9. Evaluate policy performance against baseline regret (23.32%)
10. Deploy to recovery policy engine

---

## 14. IMPLEMENTATION CHECKLIST

- [ ] Load raw CSV files (recovery_train.csv, recovery_validation.csv, recovery_test.csv)
- [ ] Classify columns per Section 2
- [ ] Remove excluded columns (recovered_amount, time_to_recovery_hours, etc.)
- [ ] Select 19 pre-intervention features + 1 action + 1 target
- [ ] Define preprocessing transformers (StandardScaler, OneHotEncoder, etc.)
- [ ] Fit transformers on training data only
- [ ] Transform all splits
- [ ] Verify no target leakage
- [ ] Verify train/val/test have matching feature dimensions after transformation
- [ ] Verify customer isolation maintained
- [ ] Document feature names and order
- [ ] Create test suite for feature pipeline
- [ ] Ready for model training

**Status:** Feature specification complete. Data ready for model training (Phase 7).
