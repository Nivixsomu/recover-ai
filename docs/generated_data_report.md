# Generated Synthetic Recovery Data Report

This report describes a controlled synthetic experimental dataset. It is **not evidence of Razorpay, bank, merchant, or customer production performance**. The UCI Online Retail archive was not read by this generator; it remains an optional behavioral reference only. UCI Credit Card Default was not used.

## Reproducibility

- Simulator: `recovery-simulator-v2`
- Random seed: `20260401`
- Python generation command: `python scripts/generate_recovery_data.py`
- Opportunities/payments: 20,000
- Customers: 4,000
- Currency: INR
- Potential-outcome rows: 100,000 (simulator-only counterfactual ground truth)

## Baseline-policy configuration

- Maximum retry count before review: 2
- Autonomous action amount limit: INR 25,000
- Human-review amount threshold: INR 50,000
- Cooldown: 24 hours since prior payment, unless this is the first retry
- Repeated historical failures requiring review: 20 or more

## Overall outcomes

- Recovery rate: **53.88%**
- Total simulated recovered revenue: **INR 22,323,775.00**
- Total amount at risk: INR 40,770,701.27
- Mean amount at risk: INR 2,038.54

## Failure distribution

| failure_reason | cases | percentage |
| --- | --- | --- |
| INSUFFICIENT_FUNDS | 2599 | 13.0 |
| PAYMENT_METHOD_DECLINED | 2519 | 12.6 |
| BANK_TIMEOUT | 2162 | 10.81 |
| NETWORK_ERROR | 2015 | 10.08 |
| TEMPORARY_BANK_ERROR | 1994 | 9.97 |
| CARD_EXPIRED | 1970 | 9.85 |
| INVALID_PAYMENT_DETAILS | 1913 | 9.56 |
| AMBIGUOUS_FAILURE | 1687 | 8.43 |
| GATEWAY_TIMEOUT | 1592 | 7.96 |
| CHECKOUT_ABANDONED | 1549 | 7.74 |

## Payment-method distribution

| payment_method | cases |
| --- | --- |
| UPI | 8858 |
| CARD | 6522 |
| NETBANKING | 2985 |
| WALLET | 1635 |

## Action distribution

| observed_action | cases | percentage |
| --- | --- | --- |
| PAYMENT_LINK | 7392 | 36.96 |
| RETRY | 7328 | 36.64 |
| NO_ACTION | 2537 | 12.68 |
| REMINDER | 1904 | 9.52 |
| HUMAN_REVIEW | 839 | 4.2 |

## Amount statistics (INR)

```text
count    20000.00
mean      2038.54
std       2902.15
min        100.00
25%        603.70
50%       1194.46
75%       2338.95
max      99223.51
```

## Customer behavior at decision time

- Mean historical transaction count: 51.74
- Mean historical success rate: 69.88%
- Mean customer lifetime value: INR 71,286.74
- Customer segments: {'NORMAL': 6722, 'RELIABLE': 6175, 'OCCASIONAL_FAILURE': 4408, 'HIGH_VALUE': 1657, 'LOW_FREQUENCY': 1038}

## Recovery outcomes by failure type

| failure_reason | cases | recovery_rate | recovered_revenue |
| --- | --- | --- | --- |
| AMBIGUOUS_FAILURE | 1687 | 38.59 | 1233681.57 |
| BANK_TIMEOUT | 2162 | 63.88 | 2733711.16 |
| CARD_EXPIRED | 1970 | 60.1 | 2486107.07 |
| CHECKOUT_ABANDONED | 1549 | 58.17 | 1948778.62 |
| GATEWAY_TIMEOUT | 1592 | 60.68 | 1892188.83 |
| INSUFFICIENT_FUNDS | 2599 | 48.86 | 2913802.78 |
| INVALID_PAYMENT_DETAILS | 1913 | 64.72 | 2519370.97 |
| NETWORK_ERROR | 2015 | 60.4 | 2409241.08 |
| PAYMENT_METHOD_DECLINED | 2519 | 33.23 | 1795421.39 |
| TEMPORARY_BANK_ERROR | 1994 | 56.72 | 2391471.53 |

## Recovery outcomes by observed action

| observed_action | cases | recovery_rate | recovered_revenue |
| --- | --- | --- | --- |
| HUMAN_REVIEW | 839 | 77.71 | 1360052.91 |
| NO_ACTION | 2537 | 6.74 | 307610.59 |
| PAYMENT_LINK | 7392 | 55.37 | 8859182.57 |
| REMINDER | 1904 | 62.87 | 2543788.41 |
| RETRY | 7328 | 63.63 | 9253140.52 |

## Leakage-safe partitions

| split | cases | customers | earliest | latest |
| --- | --- | --- | --- | --- |
| test | 3000 | 600 | 2025-11-16 00:48:40 | 2025-12-31 23:55:11 |
| train | 14000 | 2800 | 2025-01-01 00:14:34 | 2025-09-30 23:59:32 |
| validation | 3000 | 600 | 2025-10-01 00:32:44 | 2025-11-15 23:39:45 |

Partition boundaries are train: 2025-01-01T00:00:00 through 2025-09-30T23:59:59; validation: 2025-10-01T00:00:00 through 2025-11-15T23:59:59; test: 2025-11-16T00:00:00 through 2025-12-31T23:59:59. Customers are assigned to exactly one partition. Feature snapshots use history strictly before each `created_at`; outcome and potential-outcome columns are excluded from the explicit prediction feature matrix.

## Data quality and validation

- Duplicate payment IDs: 0
- Duplicate recovery case IDs: 0
- Missing values excluding expected null `time_to_recovery_hours` on failed recoveries: None
- Expected null `time_to_recovery_hours` values for unsuccessful recoveries: 9224
- Validation: **passed** — identifiers, references, timestamps, categories, amounts, outcomes, potential-outcome linkage, chronology, partition boundaries, customer separation, and leakage-column checks all passed.
