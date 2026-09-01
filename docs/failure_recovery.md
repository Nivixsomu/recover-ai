# RecoverAI — Graceful Failure & Safety Demonstration

**Document Version:** 1.0.0  
**Phase:** 14 — Failure Demonstration  
**Author:** RecoverAI Core Engineering Team  
**Status:** **ACTIVE & VALIDATED**

---

## 1. Overview

In enterprise payment infrastructure, an AI system that executes blindly without operational boundaries poses critical risks: gateway penalty fees, card network throttling, duplicate charges, and runaway operational expenses.

RecoverAI solves this by enforcing a **decoupled, multi-stage execution pipeline**:

```
ML Model (Ranking) ──► PolicyEngine (Safety Rules) ──► Razorpay Adapter (Test Mode) ──► Immutable AuditLog
```

This document demonstrates the 5 key failure modes and how RecoverAI handles them deterministically and gracefully.

---

## 2. Failure Scenarios & Validations

### Scenario A: Automatic Retry Limit Exceeded (`ERR_RETRY_LIMIT_EXCEEDED`)
* **Problem:** Repeated automated retries on persistent bank issues degrade merchant success metrics and risk issuer blocks.
* **Input Context:** `failure_reason = 'GATEWAY_TIMEOUT'`, `retry_count = 2` (Configured limit: 2).
* **ML Raw Output:** ML model predicts high probability for `RETRY` (64.2%).
* **PolicyEngine Action:**
  * Detects `retry_count >= 2`.
  * **BLOCKS `RETRY`** with machine-readable error `ERR_RETRY_LIMIT_EXCEEDED`.
  * **Falls back** to next-highest ranked permissible action: `PAYMENT_LINK`.
* **System Result:** No redundant order/retry dispatched; customer receives an interactive Razorpay payment link.

---

### Scenario B: High Ticket Autonomy Limit Exceeded (`ERR_AMOUNT_EXCEEDS_AUTONOMY_LIMIT`)
* **Problem:** Large financial amounts (> ₹25,000) should not be automatically acted upon without merchant approval.
* **Input Context:** `amount_at_risk = ₹75,000`, `failure_reason = 'PAYMENT_METHOD_DECLINED'`.
* **ML Raw Output:** ML model ranks automated links or retries.
* **PolicyEngine Action:**
  * Detects `amount_at_risk >= HUMAN_REVIEW_AMOUNT_THRESHOLD` (₹50,000) and `amount_at_risk > AUTOMATIC_ACTION_AMOUNT_LIMIT` (₹25,000).
  * Escalates case to `HUMAN_REVIEW`.
* **System Result:** Enqueues internal ticket `TICKET-XXXX` for agent intervention without triggering automated customer charges.

---

### Scenario C: Non-Retryable Error Rejection (`ERR_NON_RETRYABLE_FAILURE`)
* **Problem:** Blindly retrying cards that have expired or failed due to invalid credentials produces guaranteed 100% failure rates.
* **Input Context:** `failure_reason = 'CARD_EXPIRED'`, `payment_method = 'CARD'`.
* **ML Raw Output:** ML candidate list includes `RETRY`.
* **PolicyEngine Action:**
  * Detects customer-action failure category.
  * **BLOCKS `RETRY`** with reason `ERR_NON_RETRYABLE_FAILURE (CARD_EXPIRED requires customer action, not backend retry)`.
  * Reroutes to `PAYMENT_LINK`.
* **System Result:** Customer is sent a secure update link, avoiding wasteful backend retries.

---

### Scenario D: Idempotency Key Duplicate Prevention (`IDEMPOTENT_SKIPPED`)
* **Problem:** Network retries from upstream clients could cause duplicate payment links or multiple order charges.
* **Input Context:** Same `idempotency_key = "idem_case_12345"` submitted twice.
* **Execution Adapter Action:**
  * Checks database and in-memory idempotency register.
  * Identifies existing execution with identical token.
  * Skips secondary execution and returns `status = 'IDEMPOTENT_SKIPPED'`.
* **System Result:** Exactly-once execution guarantee; no duplicate Razorpay API calls.

---

### Scenario E: Razorpay Test Mode Gateway Timeout / API Outage
* **Problem:** External payment gateways or webhooks may time out or return 5xx errors.
* **Input Context:** `simulate_failure = True` or Razorpay API returns HTTP 504.
* **Execution Adapter Action:**
  * Catches `RazorpayTimeoutError` or `RazorpayNetworkError`.
  * Returns structured response with `execution.status = 'FAILED'`.
  * Logs failure details into SQLite `audit_events` and `executions` tables.
* **System Result:** API returns HTTP 200 with clear error details; backend server does **not** crash.

---

## 3. Machine-Readable Policy Error Codes

| Error Code | Trigger Condition | Policy Resolution |
|---|---|---|
| `ERR_RETRY_LIMIT_EXCEEDED` | `retry_count >= MAX_AUTOMATIC_RETRIES` (2) | Reroute to `PAYMENT_LINK` / `REMINDER` |
| `ERR_COOLDOWN_NOT_MET` | Attempt within `COOLDOWN_HOURS` (24h) | Block action, defer to cooldown expiry |
| `ERR_AMOUNT_EXCEEDS_AUTONOMY_LIMIT` | `amount > AUTOMATIC_ACTION_AMOUNT_LIMIT` (₹25k) | Escalate to `HUMAN_REVIEW` |
| `ERR_NON_RETRYABLE_FAILURE` | Reason in `CUSTOMER_ACTION_FAILURES` | Block `RETRY`, route to `PAYMENT_LINK` |
| `ERR_HUMAN_REVIEW_THRESHOLD_NOT_MET` | Low value ticket without escalation flags | Block `HUMAN_REVIEW`, fallback to automated |
| `ERR_INVALID_ACTION` | Action not in supported 5 actions | Rejection |

---

**FAILURE RECOVERY STATUS: FULLY DEMONSTRATED & TESTED.**
