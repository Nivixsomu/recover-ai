# RecoverAI Synthetic Recovery Data Dictionary

These controlled synthetic datasets are an experimental environment, not records of real customers, Razorpay activity, or known real-world counterfactual outcomes.

## Main observed recovery dataset

| Field | Meaning / type | Available | Role | Allowed values / range | Leakage risk |
| --- | --- | --- | --- | --- | --- |
| `recovery_case_id` | Synthetic opportunity ID, string | t0 | Identifier | Unique `RC-*` | Do not model directly. |
| `payment_id` | Synthetic payment ID, string | t0 | Identifier | Unique `PAY-*` | Do not model directly. |
| `customer_id` | Synthetic customer ID, string | t0 | Identifier/join | `CUST-*` | Do not model directly. |
| `customer_created_at` | Synthetic account start timestamp | Before t0 | Context | Earlier than `created_at` | Do not derive future tenure. |
| `customer_segment` | Synthetic behavior cohort | Before t0 | Context | Five documented segments | Simulator-only; not a production feature. |
| `created_at` | Payment decision timestamp | t0 | Feature source | 2025 split windows | Must support chronological splits. |
| `amount` / `amount_at_risk` | Payment/recoverable amount, INR decimal | t0 | Feature/value | > 0, capped at 150,000 | No later settlement values. |
| `currency` | Payment currency | t0 | Context | `INR` | None in this version. |
| `payment_method` | Simulated method | t0 | Feature | UPI, CARD, NETBANKING, WALLET | Assumptions are not production rates. |
| `payment_status` | Status at decision | t0 | Context | FAILED, ABANDONED | Never a final post-action state. |
| `failure_reason` | Simulation-only taxonomy reason | t0 | Feature | Defined failure taxonomy | Not an official Razorpay mapping. |
| `attempt_count` / `retry_count` | Attempts including current / retries before t0 | t0 | Feature/policy input | Integers ≥ 1 / ≥ 0 | Must exclude current action result. |
| `previous_failure_count` / `previous_success_count` | Earlier payment counts | Before t0 | Feature | Integers ≥ 0 | Strictly historical. |
| `historical_transaction_count` | Earlier payment-attempt count | Before t0 | Feature | Integer ≥ 1 | Strictly historical. |
| `historical_success_count` / `historical_failure_count` | Earlier success/failure counts | Before t0 | Feature | Integers ≥ 0 | Strictly historical. |
| `historical_success_rate` | Earlier successes ÷ earlier attempts | Before t0 | Feature | 0–1 | Excludes current/future outcome. |
| `historical_average_amount` | Earlier average amount, INR | Before t0 | Feature | > 0 | Excludes current/future payments. |
| `customer_lifetime_value` | Synthetic cumulative historical value, INR | Before t0 | Feature | > 0 | Frozen at t0. |
| `time_since_previous_payment_hours` | Hours since prior payment | Before t0 | Feature | > 0 | Previous event only. |
| `customer_transaction_frequency` | Earlier transactions per active month | Before t0 | Feature | > 0 | Previous events only. |
| `hour_of_day` / `day_of_week` | Time values derived at t0 | t0 | Feature | 0–23 / 0–6 | None. |
| `last_history_event_at` | Final event used for history | Before t0 | Audit/validation | Strictly before `created_at` | Audit-only. |
| `cooldown_eligible` / `amount_limit_eligible` / `human_review_required` | Baseline policy checks | t0 | Policy input | Boolean | Configuration-derived, never outcome-derived. |
| `action_allowed` | Selected baseline action is not `NO_ACTION` | t0 | Policy context | Boolean | Not an outcome. |
| `observed_action` | Baseline-selected action | At intervention | Treatment | Five recovery actions | Not a pre-action feature. |
| `recovery_success` | Simulated observed-action outcome | After intervention | Outcome | 0 or 1 | Never a feature. |
| `recovered_amount` | Simulated recovered value | After intervention | Outcome | 0 to amount at risk | Never a feature. |
| `time_to_recovery_hours` | Simulated successful delay | After intervention | Outcome | Positive or null on failure | Never a feature. |
| `split` | Evaluation partition | Generation | Metadata | train, validation, test | Never a feature. |

## Simulator-only potential-outcomes dataset

| Field | Meaning / type | Available | Role | Allowed values / range | Leakage risk |
| --- | --- | --- | --- | --- | --- |
| `recovery_case_id`, `payment_id`, `customer_id` | Synthetic join identifiers | Simulator only | Join keys | Correspond to observed row | Never raw features. |
| `potential_action` | Candidate action | Simulator only | Counterfactual treatment | Five recovery actions | Not observed real-world action. |
| `potential_recovery_probability` | Designed conditional probability | Simulator only | Ground truth | 0.01–0.95 | Never a feature. |
| `potential_recovery_success` | Sampled candidate-action outcome | Simulator only | Ground truth | 0 or 1 | Never a feature. |
| `potential_recovered_amount` | Value under candidate action | Simulator only | Ground truth | 0 to amount at risk | Never a feature. |
| `potential_time_to_recovery_hours` | Delay under candidate action | Simulator only | Ground truth | Positive or null | Never a feature. |
| `ground_truth_type` | Provenance marker | Simulator only | Metadata | `SIMULATOR_ONLY_COUNTERFACTUAL` | Prevents false real-world interpretation. |

## Customer master dataset

`recovery_customers.csv` contains `customer_id`, `customer_created_at`, `customer_segment`, historical transaction/success/failure counts and rate, `historical_average_amount`, `customer_lifetime_value`, and `simulation_cohort`. All values are synthetic and no sensitive personal attributes are generated.
