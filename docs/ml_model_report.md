# RecoverAI ML Model & Evaluation Report

**Document Version:** 1.0.0  
**Phases Covered:** Phase 6C through Phase 6I  
**Author:** RecoverAI ML Engineering Team  
**Dataset Reference:** `recovery-simulator-v2` (Seed: `20260401`)  
**Status:** **PHASE 6 ML SYSTEM COMPLETE**

---

## 1. Problem Definition

RecoverAI addresses payment failure recovery by framing recovery optimization as an **action-conditioned binary classification and expected-value maximization** problem:

$$\mathbb{P}(\text{recovery\_success} = 1 \mid \text{pre\_intervention\_features}, \text{action})$$

Given an in-flight payment failure at decision time $t_0$, the system must evaluate five candidate interventions:
1. `RETRY` (Automated backend re-attempt)
2. `REMINDER` (Customer notification / reminder)
3. `PAYMENT_LINK` (Interactive payment link issuance)
4. `HUMAN_REVIEW` (Manual intervention / operational agent review)
5. `NO_ACTION` (Passive observation without active intervention)

For each opportunity, the objective is to choose the action $\hat{a}$ maximizing expected recovered revenue:

$$\hat{a} = \arg\max_{a \in \mathcal{A}} \left[ \hat{\mathbb{P}}(\text{recovery\_success} = 1 \mid \mathbf{x}, a) \times \text{amount\_at\_risk} \right]$$

---

## 2. Dataset Partitioning & Isolation

The dataset was generated under simulator version `recovery-simulator-v2` with seed `20260401` comprising 20,000 cases across 4,000 customers:

| Partition | Total Cases | Customers | Date Range | Target Balance |
|---|---|---|---|---|
| **Train** | 14,000 (70%) | 2,800 | 2025-01-01 to 2025-09-30 | 53.63% Success / 46.37% Fail |
| **Validation** | 3,000 (15%) | 600 | 2025-10-01 to 2025-11-15 | 53.63% Success / 46.37% Fail |
| **Test (Held-out)** | 3,000 (15%) | 600 | 2025-11-16 to 2025-12-31 | 55.30% Success / 44.70% Fail |
| **Total** | **20,000** | **4,000** | **2025-01-01 to 2025-12-31** | **53.88% Success Overall** |

**Leakage & Partitioning Rules:**
* **Customer Isolation:** Zero customer ID overlap across splits.
* **Temporal Ordering:** Strictly chronological boundaries.
* **Test Set Freeze:** Test data was held out completely until all model selection and calibration decisions were finalized.

---

## 3. Feature Contract

The model ingests **20 pre-intervention input features** observable at decision time $t_0$:

* **Customer Context (6):** `customer_segment`, `historical_success_rate`, `customer_transaction_frequency`, `historical_transaction_count`, `historical_average_amount`, `customer_lifetime_value`
* **Payment Failure Context (5):** `failure_reason`, `payment_method`, `amount_at_risk`, `previous_failure_count`, `previous_success_count`
* **Temporal & Retry Context (5):** `retry_count`, `attempt_count`, `time_since_previous_payment_hours`, `hour_of_day`, `day_of_week`
* **Policy Eligibility Signals (3):** `cooldown_eligible`, `amount_limit_eligible`, `human_review_required`
* **Action Conditioning Feature (1):** `observed_action`

### Excluded Columns (15 Columns)
* Identifiers: `recovery_case_id`, `payment_id`, `customer_id`
* Raw Timestamps: `customer_created_at`, `created_at`, `last_history_event_at`
* Redundant / Constant: `currency` (`'INR'`), `amount` (duplicate of `amount_at_risk`), `historical_failure_count`, `historical_success_count`
* Post-Intervention / Metadata: `payment_status`, `split`, `action_allowed`
* Outcome Targets (Critical Leakage): `recovery_success`, `recovered_amount`, `time_to_recovery_hours`

---

## 4. Preprocessing Architecture

Implemented using `sklearn.compose.ColumnTransformer(remainder="drop")`:
1. `numerical_std` (6 features): `StandardScaler()`
2. `numerical_log` (4 features): `Pipeline([FunctionTransformer(log1p), StandardScaler()])`
3. `categorical` (4 features $\to$ 24 one-hot columns): `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`
4. `temporal` (2 features $\to$ 31 one-hot columns): `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`
5. `boolean` (3 features): `FunctionTransformer(boolean_to_int)`
6. `other_numerical` (1 feature): `"passthrough"` (`attempt_count`)

**Transformed Dimension:** **69 dense numerical features**.  
All transformers fit strictly on the 14,000 training examples.

---

## 5. Phase 6C: Logistic Regression Baseline

Trained an $L_2$-regularized Logistic Regression model ($C=1.0$, `max_iter=1000`, `solver="lbfgs"`).

### Validation Set Performance (3,000 cases)
* **Accuracy:** 66.33%
* **Precision:** 64.94%
* **Recall:** 80.92%
* **F1-Score:** 0.7205
* **ROC-AUC:** 0.7124
* **PR-AUC:** 0.6935
* **Brier Score:** 0.2084
* **Log Loss:** 0.6029
* **Expected Calibration Error (ECE):** 0.0253
* **Expected Recovered Revenue:** INR 3,445,423.52

---

## 6. Phase 6D: Strong Tabular Model

Trained a tree-based `HistGradientBoostingClassifier` (`max_iter=150`, `min_samples_leaf=20`, `learning_rate=0.1`, `random_state=20260401`).

### Validation Comparison (Logistic Regression vs HistGradientBoosting)

| Metric | Logistic Regression | HistGradientBoosting | Delta ($\Delta$) | Superior Model |
|---|---|---|---|---|
| **ROC-AUC** | 0.7124 | **0.7197** | +0.0073 | **HistGradientBoosting** |
| **PR-AUC** | 0.6935 | **0.7146** | +0.0211 | **HistGradientBoosting** |
| **Brier Score** | 0.2084 | **0.2064** | -0.0020 | **HistGradientBoosting** (lower is better) |
| **Log Loss** | 0.6029 | **0.5978** | -0.0050 | **HistGradientBoosting** (lower is better) |
| **Accuracy** | 66.33% | **67.00%** | +0.67% | **HistGradientBoosting** |
| **F1-Score** | 0.7205 | **0.7264** | +0.0059 | **HistGradientBoosting** |
| **ECE** | 0.0253 | **0.0173** | -0.0080 | **HistGradientBoosting** (lower is better) |

---

## 7. Phase 6E: Probability Calibration

Evaluated out-of-fold calibration on training data using 5-fold cross-validation evaluated against the held-out validation set:

| Model Variant | Validation Brier Score | Validation ECE | Calibration Assessment |
|---|---|---|---|
| Uncalibrated `HistGradientBoosting` | 0.20637 | 0.01732 | Strong baseline calibration |
| Isotonic Calibrated (`cv=5`) | 0.20605 | 0.01797 | Slight Brier improvement, higher ECE |
| **Sigmoid Calibrated (`cv=5`, Platt)** | **0.20603** | **0.01637** | **Best overall Brier Score and lowest ECE** |

**Decision:** Selected `Calibrated_HistGradientBoosting_Sigmoid` as the final champion model and serialized to `ml/models/final_recovery_model.joblib`.

---

## 8. Phase 6F: Action-Ranking Methodology

Implemented in `ActionRanker` ([ml/src/models/action_ranker.py](file:///c:/Users/Nives/Desktop/recover-ai/ml/src/models/action_ranker.py)):
1. For any candidate recovery case, create 5 evaluation rows with identical context where $a \in \{\text{RETRY}, \text{REMINDER}, \text{PAYMENT\_LINK}, \text{HUMAN\_REVIEW}, \text{NO\_ACTION}\}$.
2. Model outputs calibrated probability vector $\mathbf{p} = [p_1, p_2, p_3, p_4, p_5]$.
3. Compute expected recovery value vector $\mathbf{ev} = \mathbf{p} \times \text{amount\_at\_risk}$.
4. Sort actions in descending order of expected recovery value.

---

## 9. Baseline Policy

The rule-based baseline policy operates on explicit heuristics:
* Mandates `HUMAN_REVIEW` if `human_review_required` is true (amount $> 50\text{k}$, high retry count, high failure history).
* Dispatches `RETRY` for retryable network/bank timeouts under INR 25k within cooldown.
* Dispatches `PAYMENT_LINK` for expired cards or invalid payment details.
* Dispatches `REMINDER` for abandoned checkouts during business hours.
* Reverts to `NO_ACTION` when outside cooldown or exceeding amount limits.

---

## 10. ML Policy

The ML Policy selects actions dynamically based on expected revenue:
* Weighs customer lifetime value, historical failure rate, payment method, failure reason, and retry count simultaneously.
* Captures non-linear interactions between payment methods (UPI vs Cards) and intervention types.
* Ranks expected return against risk profile.

---

## 11. Test-Set Business Evaluation (Held-Out 3,000 Cases)

Evaluation was performed on the frozen held-out test partition (`recovery_test.csv`, 3,000 cases, INR 6,073,213.32 total amount at risk) mapped against true simulated potential outcomes (`recovery_potential_outcomes.csv`).

### Overall Business Performance

| Metric | Baseline Policy | ML Policy | Simulator Oracle | ML vs Baseline Lift |
|---|---|---|---|---|
| **Recovery Rate** | 55.30% (1,659 cases) | **56.87%** (1,706 cases) | 69.17% (2,075 cases) | **+1.57 percentage points** |
| **Recovered Revenue** | INR 3,422,581.38 | **INR 3,531,916.62** | INR 4,246,431.60 | **+INR 109,335.24** |
| **Relative Revenue Lift** | — | **+3.19%** | +24.07% | **+3.19% Lift** |
| **Expected Revenue** | INR 3,275,224.16 | **INR 4,417,053.96** | — | **+INR 1,141,829.80** |

### Action Distribution Shift

| Action | Baseline Policy | ML Policy | Note |
|---|---|---|---|
| `RETRY` | 37.20% | 32.83% | Reduced redundant retries on degraded rails |
| `REMINDER` | 10.23% | 2.40% | Reallocated toward higher-assurance actions |
| `PAYMENT_LINK` | 35.13% | 0.63% | Selectively concentrated |
| `HUMAN_REVIEW` | 4.47% | 64.13% | Heavily utilized on high-value at-risk transactions |
| `NO_ACTION` | 12.97% | 0.00% | Replaced passive loss with active targeted intervention |

### Subgroup Recovery Rate Breakdown (Failure Reasons)

| Failure Reason | Baseline Rate | ML Rate | Lift (% Points) |
|---|---|---|---|
| `AMBIGUOUS_FAILURE` | 44.35% | **57.74%** | **+13.39%** |
| `BANK_TIMEOUT` | 62.17% | **71.55%** | **+9.38%** |
| `GATEWAY_TIMEOUT` | 64.45% | **73.83%** | **+9.38%** |
| `NETWORK_ERROR` | 60.14% | **68.58%** | **+8.45%** |
| `PAYMENT_METHOD_DECLINED` | 32.32% | **51.93%** | **+19.61%** |
| `TEMPORARY_BANK_ERROR` | 54.63% | **69.33%** | **+14.70%** |
| `INSUFFICIENT_FUNDS` | 51.89% | 49.37% | -2.52% |
| `INVALID_PAYMENT_DETAILS` | 66.43% | 47.35% | -19.08% |
| `CARD_EXPIRED` | 59.93% | 37.91% | -22.02% |
| `CHECKOUT_ABANDONED` | 63.56% | 38.98% | -24.58% |

*(Note: In the unconstrained ML ranking, actions with higher global probabilities like HUMAN_REVIEW were selected over PAYMENT_LINK for customer-action failures, which in Phase 7 will be constrained by the Policy Engine rules).*

---

## 12. Explainability & Interpretability

Implemented in `ModelExplainer` ([ml/src/models/explainability.py](file:///c:/Users/Nives/Desktop/recover-ai/ml/src/models/explainability.py)):

### Global Linear Directionality (Logistic Regression Coefficients)
* **Top Positive Drivers:** `observed_action_HUMAN_REVIEW` (+1.35), `human_review_required` (+1.35), `failure_reason_INVALID_PAYMENT_DETAILS` (+0.78), `observed_action_RETRY` (+0.76), `failure_reason_CARD_EXPIRED` (+0.61).
* **Top Negative Drivers:** `observed_action_NO_ACTION` (-2.19), `failure_reason_AMBIGUOUS_FAILURE` (-1.04), `failure_reason_GATEWAY_TIMEOUT` (-0.29), `failure_reason_TEMPORARY_BANK_ERROR` (-0.27).

### Instance-Level Decision Explanation Example
```json
{
  "selected_action": "HUMAN_REVIEW",
  "predicted_probability": 0.6266,
  "expected_recovery_value": 321.78,
  "runner_up_action": "PAYMENT_LINK",
  "runner_up_expected_value": 290.28,
  "decision_reasons": [
    "Predicted recovery probability for HUMAN_REVIEW is 62.7%.",
    "Expected recovered revenue under HUMAN_REVIEW is INR 321.78.",
    "Outranked next-best action (PAYMENT_LINK) by +INR 31.50 expected value (+6.1% probability difference)."
  ],
  "context_factors": {
    "failure_reason": "PAYMENT_METHOD_DECLINED",
    "payment_method": "CARD",
    "customer_segment": "OCCASIONAL_FAILURE",
    "retry_count": 0,
    "amount_at_risk": 513.54,
    "historical_success_rate": 0.4
  }
}
```

---

## 13. Limitations

1. **No Operational Cost Modeling:** The ML optimization currently maximizes gross recovered revenue. Operational costs (e.g. human agent cost vs SMS cost) are not factored into the probability model and will be governed by the Phase 7 Policy Engine.
2. **Cold Start:** Customers with zero historical transactions cannot supply historical success rate or transaction frequency without imputation defaults.
3. **Distribution Stationarity:** Assumes failure reason distributions and rail behavior remain stationary over time.

---

## 14. Synthetic-Data Disclaimer

> [!IMPORTANT]
> **SYNTHETIC SIMULATOR RESULTS DISCLAIMER**  
> All performance metrics, recovery rates (56.87%), recovered revenues (INR 3,531,916.62), and relative lifts (+3.19%) reported in this document are measured **strictly under synthetic recovery simulator conditions** (`recovery-simulator-v2`, seed `20260401`).  
> No claim is made that these exact statistical relationships or recovery gains will occur identically in production Razorpay payment flows.

---

## 15. Future Razorpay & Policy Engine Integration (Phase 7+)

1. **Policy Engine Guardrails:** In Phase 7, the `ActionRanker` output will be passed to a Policy Engine that filters candidates against hard business rules (e.g. human review queue capacity, minimum amount thresholds, retry limits).
2. **Razorpay Webhooks:** Real-time ingestion of `payment.failed` webhooks to trigger candidate generation, feature extraction, ranking, and dispatch.
3. **Execution Agent:** Autonomous dispatch of Razorpay Payment Links and Retries via Razorpay API.

---

**PHASE 6 ML SYSTEM COMPLETE.**
