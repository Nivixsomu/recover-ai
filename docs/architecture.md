# RecoverAI — Complete System Architecture

**Document Version:** 2.0.0  
**Status:** **FULLY IMPLEMENTED & VALIDATED**  
**Track:** Razorpay AI Buildathon — Track 03: AI Revenue Recovery  

---

## 1. System Overview & Architecture Diagram

RecoverAI is an autonomous, policy-bounded revenue recovery engine designed to diagnose payment failure opportunities, estimate counterfactual recovery probabilities across 5 distinct actions, filter decisions through deterministic business guardrails, and execute actions via Razorpay Test Mode with a full immutable audit trail.

```
                    ┌───────────────────────────────────────────────┐
                    │               Frontend Dashboard              │
                    │      (HTML5 / CSS3 / Vanilla JS Interface)    │
                    └───────────────────────┬───────────────────────┘
                                            │ HTTP REST / JSON
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │            FastAPI Backend Service            │
                    │        (/api/v1/recovery/*, /metrics/*)       │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │           Recovery Execution Service          │
                    │         (Lifecycle Coordination Layer)        │
                    └───────┬───────────────┬───────────────┬───────┘
                            │               │               │
          ┌─────────────────┴─────┐   ┌─────┴────────┐   ┌──┴──────────────────┐
          ▼                       ▼   ▼              ▼   ▼                     ▼
   ┌──────────────┐       ┌──────────────┐    ┌──────────────┐         ┌──────────────┐
   │ Feature Eng. │  ──►  │ ML Classifier│ ─► │ ActionRanker │   ──►   │ PolicyEngine │
   │ (20 In/69 Out│       │ (Calibrated) │    │ (5 Candidates│         │ (Safety Gating
   └──────────────┘       └──────────────┘    └──────────────┘         └──────┬───────┘
                                                                              │
                                                 ┌────────────────────────────┴──────────┐
                                                 │ If allowed & execute=True             │ Always
                                                 ▼                                       ▼
                                      ┌────────────────────┐                   ┌────────────────────┐
                                      │  Razorpay Adapter  │                   │ Immutable Audit Log│
                                      │ (Test Mode/Dry Run)│                   │    (SQLite DB)     │
                                      └────────────────────┘                   └────────────────────┘
```

---

## 2. Core Subsystems & Responsibilities

### A. Feature Engineering (`ml/src/features/build_features.py`)
* **Contract:** 20 pre-intervention input features (19 transaction/customer attributes + 1 candidate action).
* **Pipeline:** Train-fitted `ColumnTransformer` expanding into 69 transformed dimensions (StandardScaler, Log1p, OneHotEncoder, BooleanToInt).
* **Zero Leakage:** Strictly isolated from target outcomes, counterfactual states, or post-intervention signals.

### B. Machine Learning & Action Ranking (`ml/src/models/`)
* **Model:** Calibrated `HistGradientBoostingClassifier` with Sigmoid (Platt) scaling.
* **Action Candidate Matrix:** For any failed payment, constructs 5 hypothetical feature rows (`RETRY`, `REMINDER`, `PAYMENT_LINK`, `HUMAN_REVIEW`, `NO_ACTION`).
* **Objective:** Computes $\hat{\mathbb{P}}(\text{recovery\_success} \mid X, a)$ and ranks actions by gross expected revenue $\mathbb{E}[V] = \hat{\mathbb{P}} \times \text{amount\_at\_risk}$.
* **What ML Does NOT Do:** ML **NEVER directly executes financial transactions or issues payment links**.

### C. Policy Engine (`ml/src/policy/policy_engine.py`)
* **Decoupled Business Guardrails:** Independent from ML weights.
* **Operational Limits:**
  * `MAX_AUTOMATIC_RETRIES` (2 retries max).
  * `AUTOMATIC_ACTION_AMOUNT_LIMIT` (₹25,000 max automatic recovery).
  * `HUMAN_REVIEW_AMOUNT_THRESHOLD` (Escalates cases $\ge$ ₹50,000).
  * `COOLDOWN_HOURS` (24h cooldown between repeated interventions).
  * Customer-Action Failure Routing (Blocks retry on `CARD_EXPIRED`, `INSUFFICIENT_FUNDS`).
* **Deterministic Fallbacks:** When the top ML action violates policy, the PolicyEngine selects the next-highest ranked permitted action and attaches machine-readable rejection reasons.

### D. Execution Adapter & Razorpay Test Mode (`backend/app/integrations/razorpay/`)
* **Test Mode Isolation:** Reads `RAZORPAY_KEY_ID` and `RAZORPAY_KEY_SECRET` from environment variables without hardcoding, logging, or leaking secrets.
* **Idempotency Protection:** Cryptographic idempotency keys prevent duplicate execution.
* **Resilience:** Graceful handling for timeouts, network outages, and mock test mode.
* **Dry Run Mode:** Supports `execute=False` to preview actions without calling external APIs.

### E. Persistent Audit Trail & Database (`backend/app/db/`, `backend/app/services/audit_service.py`)
* **Storage:** Lightweight zero-configuration SQLite (`recover_ai.db`).
* **Schema:** `recovery_cases`, `predictions`, `policy_decisions`, `executions`, and `audit_events`.
* **Immutability:** Every decision step (`CASE_RECEIVED`, `ML_RANKING_EVALUATED`, `POLICY_EVALUATED`, `ACTION_EXECUTED`) is recorded with timestamps and context payloads.

---

## 3. End-to-End Latency Benchmarks

Measured on 100 benchmark iterations:
* **ML Action Ranking (5 candidates):** 84.61 ms
* **PolicyEngine Evaluation:** 0.01 ms
* **Recovery Service + Database Persistence:** 178.35 ms
* **FastAPI End-to-End HTTP Endpoint:** 205.61 ms
