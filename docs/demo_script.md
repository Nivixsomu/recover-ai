# RecoverAI — 5-Minute Buildathon Demo Script

**Track:** Razorpay AI Buildathon — Track 03: AI Revenue Recovery  
**Presenter Guide:** Live Dashboard & API Walkthrough  
**Duration:** 5:00 minutes  

---

## Timing Breakdown

### 0:00 – 0:45 | Problem & Motivation
* **Speaker:**
  > *"Every merchant on Razorpay loses 5% to 15% of revenue to payment failures. The standard industry response is either 'blindly retry everything'—which causes card penalties and bank throttling—or 'send a generic email', which has low conversion.*
  > *Today, we present **RecoverAI**: an intelligent, action-conditioned revenue recovery platform that selects the optimal, policy-bounded intervention for every failed payment in real-time."*
* **Visual:** Open Dashboard at `http://localhost:8000/`. Point to the top KPI cards showing **+₹109,335.24 Net Revenue Lift (+3.19%)** over the heuristic baseline on 3,000 held-out test cases.

---

### 0:45 – 1:45 | Action-Conditioned ML & 5 Candidate Rankings
* **Speaker:**
  > *"Rather than predicting a single passive recovery probability, RecoverAI generates **5 candidate counterfactual feature rows** for every single transaction:*
  > *1. Immediate Backend RETRY*
  > *2. Customer REMINDER*
  > *3. Interactive PAYMENT_LINK*
  > *4. HUMAN_REVIEW Escalation*
  > *5. NO_ACTION (Passive)*
  > *Our calibrated HistGradientBoosting model estimates $\mathbb{P}(\text{success} \mid X, \text{action})$ and computes the expected recovered revenue $\mathbb{E}[V] = \hat{\mathbb{P}} \times \text{amount\_at\_risk}$."*
* **Visual:** Select **Scenario A (Transient Bank Timeout)** and click **Dry Run (Recommend)**.
* **Highlight:** Show the 5-Action horizontal ranking bars where `RETRY` is predicted at 64.3% recovery probability with an expected recovery value of ₹1,609.

---

### 1:45 – 2:45 | The Policy Engine & Operational Safety
* **Speaker:**
  > *"An AI that directly triggers financial actions is dangerous. In our audit, we discovered that unconstrained ML will over-index on Human Review because human agents have high baseline conversion in data. But human labor is scarce and expensive!*
  > *That is why RecoverAI enforces a **decoupled Policy Engine** between ML and Execution."*
* **Visual 1:** Select **Scenario B (Insufficient Funds)** and click **Dry Run**.
  * **Highlight:** Point to the red blocked badge: `🚫 RETRY: ERR_NON_RETRYABLE_FAILURE (INSUFFICIENT_FUNDS requires customer action, not backend retry)`. The PolicyEngine automatically rerouted the decision to `PAYMENT_LINK`.
* **Visual 2:** Select **Scenario D (Max Retries Exceeded, retry_count = 2)**.
  * **Highlight:** Point to `🚫 RETRY: ERR_RETRY_LIMIT_EXCEEDED (retry_count 2 >= 2)`. Policy safely falls back to `PAYMENT_LINK`.

---

### 2:45 – 3:45 | Razorpay Test Mode Execution & Idempotency
* **Speaker:**
  > *"When an action is approved by policy, RecoverAI dispatches the action through our isolated Razorpay Test Mode adapter.*
  > *Every execution is strictly guarded by cryptographic idempotency tokens to prevent duplicate charges or spamming customers."*
* **Visual:**
  1. Click **Execute (Razorpay Test Mode)**.
  2. Show the generated Razorpay Payment Link `https://rzp.io/i/mock_...` and Order ID.
  3. Toggle **Simulate Gateway Failure** and click **Execute**. Show how the system returns a structured `FAILED` response without 500 crashes.

---

### 3:45 – 4:30 | Immutable Audit Trail & Explainability
* **Speaker:**
  > *"For financial compliance, every single step in the decision lifecycle is persisted in an immutable audit ledger.*
  > *We log the exact input features, the 5 predicted probabilities, the PolicyEngine rule evaluations, the execution status, and human-readable decision explanations."*
* **Visual:** Scroll to the **Immutable Audit Trail** timeline showing `CASE_RECEIVED` $\to$ `ML_RANKING_EVALUATED` $\to$ `POLICY_EVALUATED` $\to$ `ACTION_EXECUTED`.

---

### 4:30 – 5:00 | Conclusion & Impact
* **Speaker:**
  > *"In summary, RecoverAI combines calibrated ML expected-value optimization with deterministic policy guardrails, test-mode payment execution, and comprehensive auditability.*
  > *It recovers real revenue while eliminating wasteful retries and protecting merchant reputation. Thank you!"*

---

**DEMO SCRIPT READY FOR BUILDATHON PRESENTATION.**
