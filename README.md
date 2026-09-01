# RecoverAI — Autonomous AI Revenue Recovery Platform

[![Build Status](https://img.shields.io/badge/tests-72%20passed-success)](file:///c:/Users/Nives/Desktop/recover-ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![Buildathon](https://img.shields.io/badge/Razorpay_AI_Buildathon-Track_03-indigo)](https://razorpay.com)

RecoverAI is an intelligent, policy-bounded AI revenue recovery system built for the **Razorpay AI Buildathon (Track 03: AI Revenue Recovery)**. It diagnoses failed payment opportunities, estimates counterfactual recovery probabilities across 5 actions, enforces deterministic business guardrails, and executes safe recovery actions via **Razorpay Test Mode** with an immutable audit log.

---

## 1. Problem Statement & Why Revenue Recovery Matters

E-commerce and SaaS merchants lose between **5% to 15% of gross revenue** to failed payment transactions:
* **Transient Rail Issues:** Bank timeouts, network degradation, and gateway drops.
* **Customer-Action Failures:** Expired cards, insufficient funds, or incorrect details.
* **Flawed Heuristics:** Blindly retrying every failed transaction leads to card network penalty fees, bank throttling, and poor customer experience.
* **Operational Bottlenecks:** Manual human review is expensive and unscalable without intelligent routing.

**The Solution:** An action-conditioned ML decision engine paired with an operational **Policy Engine** that determines the optimal, safest intervention for every failed payment.

---

## 2. Decoupled System Architecture

```
                    ┌───────────────────────────────────────────────┐
                    │               Frontend Dashboard              │
                    │      (Interactive Simulator & Case Review)    │
                    └───────────────────────┬───────────────────────┘
                                            │ HTTP / JSON
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │            FastAPI Backend Service            │
                    │       (/api/v1/recovery/*, /api/v1/metrics/*) │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │           Recovery Execution Service          │
                    │  (Feature Transformer -> ML -> Policy -> DB)  │
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

### Safety Principles:
1. **ML NEVER executes financial actions directly.**
2. **Every action must pass through the PolicyEngine.**
3. **Every decision is explainable, immutable, and logged.**
4. **Strictly TEST MODE ONLY — zero real-money movement.**

---

## 3. ML Approach & Feature Pipeline

* **20 Pre-Intervention Input Features:** 19 transaction/customer/failure attributes + 1 candidate action (`observed_action`).
* **69 Transformed Dimensions:** Train-fitted scaling (`StandardScaler`), log-transforms (`log1p`), one-hot encodings, and boolean integer mappings.
* **Champion Model:** `HistGradientBoostingClassifier` with **Sigmoid (Platt) probability calibration** (`Brier Score = 0.2060`, `ECE = 0.0164`).
* **Action Candidate Matrix:** Expands any single transaction into 5 candidate feature vectors: `RETRY`, `REMINDER`, `PAYMENT_LINK`, `HUMAN_REVIEW`, `NO_ACTION`.
* **Expected-Value Optimization:**
  $$\mathbb{E}[V(X, a)] = \hat{\mathbb{P}}(\text{recovery\_success} \mid X, a) \times \text{amount\_at\_risk}$$

---

## 4. Policy Engine & Safety Guardrails

The Policy Engine enforces deterministic business rules on ML recommendations:
* **Retry Limit (`ERR_RETRY_LIMIT_EXCEEDED`):** Maximum 2 automatic retries.
* **Cooldown Guardrail (`ERR_COOLDOWN_NOT_MET`):** Prohibits actions within 24h of previous attempt.
* **Autonomy Amount Cap (`ERR_AMOUNT_EXCEEDS_AUTONOMY_LIMIT`):** Automatically routes tickets > ₹25,000 to `HUMAN_REVIEW`.
* **Customer Error Rerouting (`ERR_NON_RETRYABLE_FAILURE`):** Blocks `RETRY` on `CARD_EXPIRED` or `INSUFFICIENT_FUNDS` and reroutes to `PAYMENT_LINK`.
* **Human Review Escalation:** Low-value tickets without escalation criteria are blocked from expensive human labor and fall back to automated digital actions.

---

## 5. Measured Evaluation & Business Results

Evaluated on the 3,000 held-out test cohort from `recovery-simulator-v2` (Seed: `20260401`):

| Metric | Baseline Heuristic | RecoverAI ML + Policy | Net Lift |
|---|---|---|---|
| **Total Amount at Risk** | ₹6,073,213.32 | ₹6,073,213.32 | — |
| **Recovered Revenue** | ₹3,422,581.38 | **₹3,531,916.62** | **+₹109,335.24** |
| **Recovery Rate** | 55.30% | **56.87%** | **+1.57% points** |
| **Relative Revenue Gain** | — | — | **+3.19% lift** |
| **Oracle Potential Ceiling** | — | ₹4,246,431.60 (69.17%) | — |

> [!NOTE]
> **Synthetic Data Disclaimer:** These figures reflect controlled, counterfactual simulation dynamics in `recovery-simulator-v2` and do not represent live production merchant results.

---

## 6. Installation & Quickstart

### Prerequisites
* Python 3.12+
* Windows / macOS / Linux

### 1. Environment Setup
```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt
```

### 2. Environment Variables (.env)
Create a `.env` file from the template (optional for mock test mode, required for live Razorpay Test API):
```env
RAZORPAY_KEY_ID=rzp_test_YourTestKeyIdHere
RAZORPAY_KEY_SECRET=YourTestKeySecretHere
```

### 3. Run FastAPI Backend & Web Dashboard
```powershell
uvicorn backend.app.main:app --reload --port 8000
```
* **Web Dashboard:** Open `http://localhost:8000/` in your browser.
* **Interactive OpenAPI Docs:** Open `http://localhost:8000/docs`.

### 4. Run Test Suite
```powershell
python -m pytest
```
* **Test Count:** **72 / 72 passed (100%)**.

---

## 7. Demo Guide (5-Minute Walkthrough)

1. **Open Dashboard (`http://localhost:8000`):** Review top-level KPIs and synthetic lift metrics.
2. **Scenario A (Transient Timeout):** Select "Scenario A", click **Dry Run**, and observe ML ranking `RETRY` at 64.3% probability with policy approval.
3. **Scenario B (Insufficient Funds):** Select "Scenario B" and observe the PolicyEngine blocking `RETRY` due to non-retryable customer failure, routing instead to `PAYMENT_LINK`.
4. **Scenario C (High Value ₹75k):** Select "Scenario C" and observe automatic autonomy limits routing the case to `HUMAN_REVIEW` (enqueuing `TICKET-XXXX`).
5. **Scenario D (Max Retries):** Select "Scenario D" (retry_count = 2) and observe policy blocking `RETRY` (`ERR_RETRY_LIMIT_EXCEEDED`) and falling back to `PAYMENT_LINK`.
6. **Execution & Audit Trail:** Click **Execute (Razorpay Test Mode)** to generate a live test order/link and inspect the chronological SQLite audit trail.

---

## 8. Failure Recovery Scenarios

Full failure scenarios and deterministic error codes are detailed in [`docs/failure_recovery.md`](docs/failure_recovery.md):
* **Duplicate Prevention:** Idempotency key tracking blocks duplicate charges (`IDEMPOTENT_SKIPPED`).
* **Gateway Resiliency:** Simulated network/timeout errors return structured `FAILED` status without crashing the API.

---

## 9. Project Structure

```
recover-ai/
├── backend/
│   ├── app/
│   │   ├── db/            # SQLite session & table schema
│   │   ├── integrations/  # Razorpay Test Mode client & exceptions
│   │   ├── routers/       # /recovery and /metrics REST endpoints
│   │   ├── schemas/       # Pydantic request/response models
│   │   ├── services/      # RecoveryService & AuditService
│   │   ├── config.py      # Environment settings
│   │   └── main.py        # FastAPI app & static file mount
│   └── tests/             # Backend, API, and E2E test suites
├── frontend/              # Modern dark mode web dashboard
│   ├── index.html
│   ├── style.css
│   └── app.js
├── ml/
│   ├── data/              # Train/validation/test partitions & UCI sources
│   ├── models/            # Trained champion model & metadata artifacts
│   ├── src/
│   │   ├── features/      # 20-to-69 dimension feature pipeline
│   │   ├── models/        # Training, ActionRanker & explainability
│   │   └── policy/        # PolicyEngine & safety rules
│   └── tests/             # Feature, model, policy & simulator tests
├── docs/                  # Architectural specs, audit reports & demo guide
└── scripts/               # Training, policy audit & performance benchmarks
```

---

## 10. Known Limitations & Future Work

* **Simulated Counterfactuals:** True production deployment will require online multi-armed bandit (MAB) / contextual bandit exploration to update conversion priors.
* **Labor Cost Modeling:** Incorporating agent hourly wage into the objective function: $\mathbb{E}[\text{Net}] = \hat{\mathbb{P}} \times \text{Amount} - \text{Labor Cost}$.
* **Multi-Channel Dispatch:** Integrating WhatsApp Business API and SMS webhooks alongside Razorpay Payment Links.

---

## 11. Security & Compliance

* **Test Mode Only:** No real money movement is supported or permitted.
* **Zero Credential Exposure:** Secrets are read strictly from OS environment variables and never logged, returned in API payloads, or committed to Git.
* **Idempotent Dispatch:** All financial actions require cryptographic idempotency keys.
