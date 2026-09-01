# RecoverAI Controlled Recovery Simulation Design

## Implementation Status

**Current Version:** recovery-simulator-v2  
**Status:** Phase 5B Complete — Ready for ML  
**Seed:** 20260401  
**Dataset:** 20,000 recovery cases; 4,000 customers; 100,000 potential outcomes  
**Recommendation:** A. READY FOR ML

### Phase 5B Achievement

The v2 simulator successfully implements action-conditional outcome modeling with meaningful feature×action interactions. Key metrics:

- Failure-reason-only optimal-action lookup accuracy: **61.62%** (down from 100% in v1)
- Failure reasons with multiple optimal actions: **9 of 10**
- Expected recovery rate: **53.88%**
- Baseline-selected recovered revenue: INR 22,287,427.01
- Probability-optimal expected recovered revenue: INR 29,067,358.78
- Baseline relative action regret: **23.32%**

The simulator demonstrates that feature context (customer history, payment amount, retry count, payment method, time of day, and customer segment) meaningfully changes the probability-optimal recovery action within each failure reason.

### Tests Implemented

All property-based tests verify v2 design goals:
- Deterministic generation with fixed seed
- Multiple optimal actions within failure reasons based on context
- No leakage between pre-intervention features and outcomes
- Chronological train/validation/test partitions with no customer overlap
- Recovery probability bounds [0.01, 0.95]
- All five actions represented in counterfactual outcomes
- Outcome consistency (recovery_success ↔ recovered_amount)
- Action-conditional variation in recovery rates
- Feature context influences optimal action

---

## Purpose and scope

RecoverAI needs data that represents a recovery decision: a payment has failed, an intervention is selected, and a later outcome is observed. Neither investigated UCI dataset provides that complete sequence. This design therefore specifies a **controlled synthetic simulation** for research and demonstration only.

It does not assert real-world Razorpay behavior, does not create any data in this phase, and does not authorize any payment action. Every probability, threshold, and cost described below is a versioned simulation assumption to be reviewed before implementation.

```text
Customer history + payment event + failure context
                    │
                    ▼
             feature snapshot at t0
                    │
                    ▼
          candidate action / policy decision
                    │
                    ▼
 action-conditional simulated recovery outcome
                    │
                    ▼
      measurement, audit, and policy evaluation
```

## Proposed data entities

The simulator will generate internally consistent customers, chronological payment histories, and failed payment events. A row is an **intervention opportunity** at decision time `t0`, not an arbitrary independent random record.

| Entity | Initial fields | Simulation relationship |
| --- | --- | --- |
| Customer | `customer_id`, customer history, transaction frequency, historical success rate, average transaction value, customer lifetime value | Customer-level latent behavior influences their sequence of payments; aggregates at `t0` are calculated only from earlier simulated events. |
| Payment | `payment_id`, `customer_id`, amount, timestamp, payment method, payment status, failure reason | A payment belongs to one customer, has a time-ordered context, and becomes an intervention opportunity only when its status is a simulated failure. |
| Payment history | previous attempts, previous failures, recent successful payments, time since previous payment | Derived from events strictly before `t0`; it influences eligibility and simulated recoverability. |
| Recovery decision | `action` (`RETRY`, `REMINDER`, `PAYMENT_LINK`, `NO_ACTION`, `HUMAN_REVIEW`) | Assigned by an exploration policy or a defined baseline, subject to future policy gates. |
| Outcome | `recovery_success`, `recovered_amount`, `time_to_recovery` | Sampled after the action from an action-conditional outcome model; never present in the pre-intervention feature snapshot. |

`customer_id` and `payment_id` are synthetic identifiers. They support chronology and auditing but must not be used as raw predictive features.

## Failure taxonomy

The following is an initial, simulation-only taxonomy. It is deliberately not presented as official Razorpay failure semantics or API mappings.

| Failure category | Failure reasons | Initial handling class | Simulation rationale |
| --- | --- | --- | --- |
| Temporary / system | `BANK_TIMEOUT`, `GATEWAY_TIMEOUT`, `NETWORK_ERROR`, `TEMPORARY_BANK_ERROR` | Retryable | A later retry can plausibly succeed when the failure is transient, subject to retry limits and cooldowns. |
| Payment / customer details | `INSUFFICIENT_FUNDS` | Not immediately retryable; customer-action path | Immediate retry benefit is simulated as lower; a delayed reminder or payment link may be more appropriate. |
| Payment / customer details | `CARD_EXPIRED`, `INVALID_PAYMENT_DETAILS` | Requires customer action | Retrying unchanged details is not treated as an ordinary solution; payment link or human-assisted path is favored. |
| Payment / customer details | `PAYMENT_METHOD_DECLINED` | Review-dependent | May require a payment-method change, an alternative path, or human review; automatic retry eligibility is constrained. |
| Abandonment | `CHECKOUT_ABANDONED` | Customer-action path | No failed charge is assumed; reminder/payment-link style intervention is considered rather than a blind retry. |
| Ambiguous / elevated risk | future `UNKNOWN` or repeated-failure cases | Human review or no action | Used when evidence is insufficient, value is high, or automation limits are exceeded. |

The eventual simulator will maintain a versioned taxonomy/configuration table with an allowed-action set, cooldown family, and parameter ranges for every reason. New categories will default to `HUMAN_REVIEW` or `NO_ACTION` until explicitly reviewed.

## Causal and business assumptions

The simulator will create dependencies rather than independently randomizing columns:

- Customer transaction frequency, historical success rate, average transaction value, and lifetime value arise from the customer’s earlier payment sequence.
- The amount and payment method affect payment-status and failure-reason distributions; no single feature deterministically fixes an outcome.
- Temporary technical failures generally receive a higher simulated incremental benefit from `RETRY` than other actions.
- `INSUFFICIENT_FUNDS` has lower immediate retry potential and can receive relatively more benefit from a later `REMINDER` or `PAYMENT_LINK`.
- `CARD_EXPIRED` and invalid details are not treated as ordinary retry cases; a customer-action path has greater simulated relevance.
- `CHECKOUT_ABANDONED` is modeled as a reminder/payment-link opportunity rather than a payment that can blindly be retried.
- Repeated failed attempts, short gaps between attempts, and excessive prior interventions reduce automatic eligibility and simulated success probability.
- Strong prior payment success and recent successful payments can increase simulated recoverability, while long inactivity can reduce it.
- Larger amounts may have greater potential recovered value but stricter policy thresholds and a greater chance of `HUMAN_REVIEW`.

These are assumptions for controlled scenarios, not facts about merchants, customers, banks, or Razorpay. Implementation must expose them as documented parameters, use a reproducible random seed, and retain the simulation configuration version with each generated batch.

## Action-conditional outcomes

The simulation target is:

```text
P(recovery_success | features at t0, action)
```

not merely `P(recovery_success | features at t0)`. For each failed payment, the future simulator will:

1. Freeze a pre-intervention feature snapshot at `t0`.
2. Identify actions allowed by the simulated failure taxonomy and policy configuration.
3. Compute one probability per candidate action using a transparent base-rate-plus-adjustment model. Adjustments can depend on failure type, retry count, time since failure, payment amount band, payment method, and historical behavior.
4. Bound all probabilities to a reviewed range and sample the outcome only for the selected action.
5. If recovery succeeds, sample `recovered_amount` no greater than the payment amount and sample a positive `time_to_recovery` from an action/failure-specific delay distribution; otherwise set recovered amount to zero and record a defined censored/no-recovery outcome.

For example, a transient failure can have a larger `RETRY` uplift than a link-based uplift, while an expired payment method can have little retry uplift but a higher customer-action-path uplift. The eventual parameterization will be scenario-based and reviewed—not a fixed copy of illustrative numbers.

To make later action evaluation meaningful, the simulation should assign actions with controlled exploration among policy-allowed alternatives, rather than always picking the action with the highest designed probability. The chosen action and the generation-policy version must be logged. Potential outcomes for unchosen actions may be retained only as clearly marked simulator truth for offline evaluation; they must never be represented as observed real-world outcomes.

## Public-data use and boundaries

### UCI Online Retail

Online Retail may inform distributions or aggregate-design choices for transaction amounts, purchase frequency, temporal behavior, cancellation/reversal signals, and customer-level historical aggregates. Its use must remain time-bounded and must preserve the distinction between an invoice cancellation and a payment failure.

It cannot provide real payment failures, recovery interventions, contact attempts, recovery success, recovered amount, or time to recovery. Those outcomes will be simulated separately.

### UCI Default of Credit Card Clients

The credit-default dataset will **not** be merged into the main recovery dataset. It is a different domain, geography, unit of analysis, and outcome definition. It may only support separate exploratory discussion of payment/default-history concepts and leakage/fairness risks; its default label is never a recovery label.

## Feature / outcome boundary and leakage controls

| Available before intervention at `t0` | Known only after intervention |
| --- | --- |
| Payment amount, timestamp-derived features, payment method, failure reason, retry count, prior failures, recent successful payments, time since previous payment, historical success rate, transaction frequency, average value, lifetime value | `recovery_success`, `recovered_amount`, `time_to_recovery`, later payment state, later attempts, later reminders, later customer activity, action outcome/cost |

Rules for the future simulator and model pipeline:

- Construct every feature snapshot using events with timestamps strictly earlier than `t0`.
- Do not use `recovery_success`, recovered value, outcome delay, a post-action status, or any aggregate that includes the current/future event as a feature.
- Treat retry count as its value **before** selecting the current action; do not include the result of the current retry.
- Maintain immutable event timestamps and an audit record for feature-window start/end, decision time, action time, and outcome observation time.
- Keep outcome-generation code and feature-generation code separate, with tests that assert no outcome column is present in the feature set.

## Train, validation, and test design

The future dataset should be split chronologically, with an additional customer-level boundary where feasible:

1. Choose an early time window for training, a later contiguous window for validation, and the latest contiguous window for testing.
2. Build historical features in each split using only events available before each row’s decision time.
3. Prefer holding out customers from validation/test for generalization measurement; when the business question requires known-customer behavior, report a second, explicitly labeled temporal-only evaluation.
4. Fit all encoders, normalizers, thresholds, and calibration steps on the training period only.

Random row splitting is dangerous because one customer’s later transactions can enter training while an earlier transaction is in testing, causing customer-history and future-behavior leakage. It can also distribute correlated retries for one payment across splits.

## Non-ML baseline strategy

The first benchmark will be a transparent, rule-based policy; no ML model is implemented now. Its initial specification is:

1. If a temporary/system failure is present, retry count is below the configured maximum, cooldown has elapsed, amount is at or below the automatic-action limit, and no stopping rule applies: `RETRY`.
2. If the failure requires customer action (`INSUFFICIENT_FUNDS`, expired/invalid details, or abandonment) and contact eligibility/cooldown permits: `REMINDER` or `PAYMENT_LINK`, selected by a documented deterministic rule.
3. If the transaction exceeds a review threshold, failures repeat beyond a limit, a reason is ambiguous, or a policy check is unavailable: `HUMAN_REVIEW`.
4. If an action is ineligible, the cooldown is active, or intervention limits are exhausted: `NO_ACTION`.

Future ML proposals must demonstrate improvement over this baseline under the same policy constraints, costs, and batch-level evaluation—not merely a higher score on cherry-picked rows.

### Implemented synthetic v1 configuration

The Phase 4 simulator uses seed `20260401`, maximum retry count `2`, autonomous action amount limit INR 25,000, human-review amount threshold INR 50,000, a 24-hour cooldown (except a first retry), and human review after 20 or more previous failures. These are configurable synthetic baseline settings, not production policy values.

## Evaluation plan

All metrics are calculated over a complete held-out batch and reported overall, by failure reason, by action, amount band, and relevant safety cohorts.

| Metric | Definition / purpose |
| --- | --- |
| Recovery rate | Recovered payment opportunities ÷ eligible opportunities. |
| Total recovered revenue | **Σ `recovered_amount` across every evaluated opportunity in the batch**. Failed/no-recovery rows contribute zero; amounts are never counted more than once per payment. |
| Expected recovered value | Σ predicted recovery probability × allowable amount, net of modeled intervention cost where applicable. |
| Recovery rate by failure type/action | Detects policies that work only for selected reasons or actions. |
| Precision, recall, F1 | For a separately defined binary decision such as “intervene” or “likely recoverable”; report the threshold and class definition. |
| PR-AUC | Threshold-independent ranking quality for imbalanced recoverability labels. |
| Calibration | Agreement between predicted probabilities and observed/simulated frequency in held-out data. |
| False-positive intervention cost | Cost assigned when an intervention is taken without recovery, including modeled customer/operational cost. |
| Unnecessary intervention rate | Interventions with no recovery or no expected positive value ÷ all interventions, using the documented definition. |
| Policy violation count | Number of proposed/executed actions breaching retry, amount, cooldown, idempotency, or review rules; target is zero. |

For policy comparison, use the same payment opportunities, action eligibility, and outcome simulation seed/configuration. Report confidence intervals or repeated simulation runs rather than selecting a favorable batch.

## Safety and policy concepts

An eventual prediction is advisory only:

```text
ML prediction → Policy Engine → Allowed / Blocked → Action
```

The policy engine is not implemented in this phase. Its initial concepts are:

- **Maximum retry count:** restrict automatic retries per payment/reason within a defined time window.
- **Transaction amount limits:** require human review above a configurable value or when expected value/risk is uncertain.
- **Cooldown period:** prevent rapid repeat actions to the same payment/customer.
- **Human-review threshold:** route high-value, ambiguous, repeated, or policy-sensitive cases to review.
- **Idempotency:** one stable action key per payment/action/attempt so duplicate events cannot create duplicate interventions or revenue accounting.
- **Audit logging:** record feature snapshot version, model/version if later added, proposed action, policy decision and reason, execution identifier, and eventual outcome.
- **Stopping rules:** cease automation after recovery, action-limit exhaustion, customer opt-out/negative signal, policy block, or escalation to review.

No policy parameter in this document is an authorization to make a real payment, send a message, or call Razorpay.

## Open design decisions before implementation

- Define the simulated merchant context, currency display convention, amount distribution transformation, and intervention-cost assumptions.
- Review action eligibility, threshold ranges, and fairness constraints with product/domain stakeholders.
- Specify contact consent, opt-out, and communication-channel assumptions before simulating reminders.
- Decide whether simulator-only potential outcomes will be retained for counterfactual evaluation and ensure they are permanently labelled synthetic.
- Add data-schema, chronology, and leakage tests before generating any synthetic dataset.
