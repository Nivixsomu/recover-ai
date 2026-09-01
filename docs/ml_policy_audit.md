# RecoverAI — ML Policy & HUMAN_REVIEW Audit

**Document Version:** 1.0.0
**Phase:** 7 — ML Policy Audit
**Author:** RecoverAI ML Engineering Team
**Dataset Reference:** `recovery-simulator-v2` (Seed: `20260401`)
**Status:** **AUDIT COMPLETE — POLICY ENGINE REQUIRED**

---

## 1. Executive Summary & Audit Question

In Phase 6G, evaluating the unconstrained ML policy against the 3,000 held-out test cases revealed a substantial shift in the action distribution:
* **Baseline Policy:** `HUMAN_REVIEW` selected for **4.47%** of cases (134 cases).
* **Unconstrained ML Policy:** `HUMAN_REVIEW` selected for **64.13%** of cases (1,924 cases).

### Core Audit Finding
The high proportion of `HUMAN_REVIEW` recommendations by the ML model is **mathematically rational under unconstrained gross-revenue maximization**, but **operationally unviable without a Policy Engine** because:
1. **Simulator Potential Conversion:** In the simulator, manual human follow-up achieves an unconditional average success probability of **62.50%** across all failure types (higher than automated methods like generic retries at 28.68% or payment links at 40.97%).
2. **Zero Operational Cost Modeling:** The ML optimization objective maximizes pure expected gross revenue E[V] = P(success) * amount_at_risk with no cost penalty for human labor (e.g., INR 100-300 per ticket).
3. **Absence of Capacity Constraints:** The unconstrained model assigns `HUMAN_REVIEW` to any ticket where human intervention yields even a +1% marginal conversion gain over automated actions.

**Conclusion:** The ML model is functioning accurately as an expected gross value estimator. However, **ML must never execute actions directly**. A separate, decoupled **Policy Engine** is strictly necessary to enforce operational capacity, amount thresholds, retry limits, and business constraints before dispatching actions.

---

## 2. Action Distribution Comparison (Test Set)

| Recovery Action | Baseline Heuristic (%) | Unconstrained ML Policy (%) | Net Shift (% Points) |
|---|---|---|---|
| `HUMAN_REVIEW` | 4.47% (134) | **64.13% (1,924)** | **+59.66%** |
| `RETRY` | 37.20% (1,116) | **32.83% (985)** | -4.37% |
| `REMINDER` | 10.23% (307) | **2.40% (72)** | -7.83% |
| `PAYMENT_LINK` | 35.13% (1,054) | **0.63% (19)** | -34.50% |
| `NO_ACTION` | 12.97% (389) | **0.00% (0)** | -12.97% |

---

## 3. Simulator Potential Conversion Rates Across Actions

The synthetic simulator ground-truth potential outcomes across all 3,000 test cases (evaluating all 5 actions counterfactually per case = 15,000 evaluations) show why the ML model learned to prefer `HUMAN_REVIEW`:

| Potential Action | Mean Simulated Probability | Median Probability | True Counterfactual Success Rate |
|---|---|---|---|
| `HUMAN_REVIEW` | **62.50%** | **61.80%** | **60.70%** |
| `PAYMENT_LINK` | 40.97% | 36.81% | 42.43% |
| `REMINDER` | 39.06% | 38.33% | 39.07% |
| `RETRY` | 28.68% | 12.03% | 28.47% |
| `NO_ACTION` | 7.01% | 7.08% | 7.03% |

Because `HUMAN_REVIEW` exhibits the highest unconditional baseline success across diverse failure categories in the synthetic simulator, an algorithm optimizing solely for P(success) * amount will inevitably allocate the majority of cases to `HUMAN_REVIEW` whenever automated actions have lower probability.

---

## 4. Breakdown of ML Action Allocations

### A. By Failure Reason (%)
| failure_reason | HUMAN_REVIEW | PAYMENT_LINK | REMINDER | RETRY |
| --- | --- | --- | --- | --- |
| AMBIGUOUS_FAILURE | 75.73 | 0.42 | 4.18 | 19.67 |
| BANK_TIMEOUT | 56.30 | 0.59 | 3.52 | 39.59 |
| CARD_EXPIRED | 60.65 | 0.72 | 1.81 | 36.82 |
| CHECKOUT_ABANDONED | 64.83 | 0.42 | 4.24 | 30.51 |
| GATEWAY_TIMEOUT | 59.77 | 0.00 | 3.52 | 36.72 |
| INSUFFICIENT_FUNDS | 69.27 | 0.50 | 1.01 | 29.22 |
| INVALID_PAYMENT_DETAILS | 61.13 | 1.77 | 1.77 | 35.34 |
| NETWORK_ERROR | 59.80 | 1.35 | 3.38 | 35.47 |
| PAYMENT_METHOD_DECLINED | 66.30 | 0.55 | 0.55 | 32.60 |
| TEMPORARY_BANK_ERROR | 67.73 | 0.00 | 1.60 | 30.67 |

### B. By Amount Band (%)
| amount_band | HUMAN_REVIEW | PAYMENT_LINK | REMINDER | RETRY |
| --- | --- | --- | --- | --- |
| INR_1001_5000 | 63.08 | 0.54 | 2.22 | 34.16 |
| INR_100_1000 | 62.66 | 0.55 | 2.69 | 34.10 |
| INR_25001_PLUS | 100.00 | 0.00 | 0.00 | 0.00 |
| INR_5001_25000 | 77.37 | 1.65 | 2.06 | 18.93 |

### C. By Retry Count (%)
| retry_count | HUMAN_REVIEW | PAYMENT_LINK | REMINDER | RETRY |
| --- | --- | --- | --- | --- |
| 0.00 | 56.85 | 0.77 | 2.67 | 39.72 |
| 1.00 | 97.72 | 0.00 | 1.71 | 0.57 |
| 2.00 | 100.00 | 0.00 | 0.00 | 0.00 |
| 3.00 | 100.00 | 0.00 | 0.00 | 0.00 |

---

## 5. Architectural Remedies & Policy Engine Requirements

To transform this unconstrained ML recommendation into an enterprise-grade, cost-effective recovery system, the **Policy Engine** must enforce the following deterministic safety guardrails:

1. **`HUMAN_REVIEW` Thresholds:**
   * Mandated only when `amount_at_risk >= HUMAN_REVIEW_AMOUNT_THRESHOLD` (e.g. INR 50,000), or `retry_count >= 3`, or `previous_failure_count >= 20`, or `failure_reason == 'AMBIGUOUS_FAILURE'`, or explicit policy flags.
   * For low-to-medium value customer-action failures, `HUMAN_REVIEW` is **blocked by policy**, and the Policy Engine automatically **falls back to the next-highest ranked automated action** (`PAYMENT_LINK` or `REMINDER`).

2. **`RETRY` Guardrails:**
   * Prohibited if `retry_count >= MAX_AUTOMATIC_RETRIES` (max 2 retries).
   * Prohibited if within `cooldown_hours` (< 24h since previous attempt).
   * Prohibited if `amount_at_risk > AUTOMATIC_ACTION_AMOUNT_LIMIT` (INR 25,000).

3. **`PAYMENT_LINK` & `REMINDER` Guardrails:**
   * Automated customer contact actions are prioritized for non-retryable customer errors (`CARD_EXPIRED`, `INVALID_PAYMENT_DETAILS`, `INSUFFICIENT_FUNDS`).
   * Subject to rate-limiting and duplicate link prevention.

4. **Machine-Readable Explanations:**
   * Every blocked action must record an immutable, auditable policy reason code (e.g. `ERR_POLICY_AMOUNT_EXCEEDS_AUTONOMY`, `ERR_POLICY_RETRY_LIMIT_EXCEEDED`, `ERR_POLICY_HUMAN_REVIEW_CAPACITY`).

---

**AUDIT STATUS: COMPLETE — PROCEEDING TO POLICY ENGINE IMPLEMENTATION.**