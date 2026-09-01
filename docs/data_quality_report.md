# RecoverAI Data Quality Report

Inspection date: 29 August 2026. The read-only script `scripts/inspect_datasets.py` loaded each original Excel workbook directly from its UCI ZIP archive; no source data was extracted, edited, merged, or labeled.

## UCI Online Retail

**Observed shape:** 541,909 rows × 8 columns. **Duplicate full rows:** 5,268 (0.97%). **Date range:** 2010-12-01 08:26:00 to 2011-12-09 12:50:00. **Unique non-null customers:** 4,372.

Although the UCI metadata says no missing values, the workbook itself contains 1,454 missing `Description` values (0.27%) and 135,080 missing `CustomerID` values (24.93%). The latter limits customer-level analysis to identified customers and must not be silently imputed as a customer identity.

`Quantity`, `UnitPrice`, and the derived in-memory line amount (`Quantity × UnitPrice`) have substantial negative and extreme values. For example, quantity ranges from -80,995 to 80,995, unit price from -11,062.06 to 38,970.00, and line amount from -168,469.60 to 168,469.60. These require event semantics and outlier investigation before any future use.

| Column | Meaning and observed type | Missing | RecoverAI usefulness | Safe prediction feature? | Outcome / label status | Leakage risk |
| --- | --- | ---: | --- | --- | --- | --- |
| `InvoiceNo` | Transaction invoice identifier; `C` prefix means cancellation. `object` | 0.00% | Useful to identify transaction and cancellation events during exploration. | No; an identifier. | Not a recovery label; cancellation state is an observed transaction status. | High if its cancellation prefix or a later invoice state is used to predict an earlier decision. |
| `StockCode` | Product/item code. `object` | 0.00% | May support product-level transaction-pattern exploration. | Conditionally, only with time-bounded encoding and unseen-product handling. | Not a label. | Medium; high cardinality and future frequency/target encodings can leak. |
| `Description` | Product name. `object` | 0.27% | Useful for manual product interpretation. | Generally no at this stage; high-cardinality/free-text field. | Not a label. | Medium; text-derived aggregates must exclude future data. |
| `Quantity` | Units in the invoice line. `int64` | 0.00% | Useful to describe purchase/cancellation magnitude. | Conditionally, when known at the event decision time. | Not a label; negative values can reflect reversals/cancellations. | High if a later adjustment is represented in a row used for an earlier event. |
| `InvoiceDate` | Invoice event date and time. `datetime64[us]` | 0.00% | Essential for chronological investigation and future time-based splits. | Yes, only for features available before the prediction cutoff. | Not a label. | High if random splits or future aggregates are used. |
| `UnitPrice` | Per-item price in sterling. `float64` | 0.00% | Useful for value and amount-distribution analysis. | Conditionally, if available when an action would be decided. | Not a label. | Medium; later corrections or invoice reversals can contaminate timing. |
| `CustomerID` | Customer identifier. `float64` because of nulls | 24.93% | Supports historical behavior analysis for identified customers. | No as a raw feature; use only future time-bounded aggregates after a privacy review. | Not a label. | Very high: customer history that includes future transactions leaks; missingness may also bias cohorts. |
| `Country` | Customer residence country. `str` | 0.00% | Useful for geographic distribution and operational segmentation review. | Conditionally, after fairness, necessity, and stability review. | Not a label. | Low temporal leakage, but potential proxy/fairness risk and limited geographic generalization. |

**Important distributions:** United Kingdom accounts for 495,478 rows (91.43%). The median quantity is 3, median unit price is £2.08, and median derived line amount is £9.75. These are transaction-line records, not unique orders or customers.

**Decision:** Use only for investigation and eventual simulation design. It should not be used as an actual recovery-outcome training dataset because it contains no recovery action, contact, payment-failure, or recovered-amount ground truth.

## UCI Default of Credit Card Clients

**Observed shape:** 30,000 rows × 25 columns. **Duplicate full rows:** 0. **Unique `ID` values:** 30,000. **Missing values:** 0 in every column. There is no date column, so a full event-time analysis or chronological train/test split cannot be constructed from this workbook alone.

The observed response distribution is 23,364 non-default records (77.88%) and 6,636 default records (22.12%). This is a **default-payment** outcome, not a recovery-success or recovery-action outcome.

| Column(s) | Meaning and observed type | Missing | RecoverAI usefulness | Safe prediction feature? | Outcome / label status | Leakage risk |
| --- | --- | ---: | --- | --- | --- | --- |
| `ID` | Client record identifier. `int64` | 0.00% | Row identity and uniqueness checking only. | No. | Not a label. | High if an arbitrary identifier is learned. |
| `LIMIT_BAL` | Given credit limit in NT dollars. `int64` | 0.00% | Potential exploratory measure of account scale. | Only conditionally for a future default-risk research exercise; not an action feature without governance. | Not a label. | Must be verified as available before the target period. |
| `SEX` | Encoded gender. `int64` | 0.00% | Fairness auditing only. | No for automated recovery/action decisions pending legal and fairness review. | Not a label. | Sensitive-attribute and disparate-impact risk. |
| `EDUCATION` | Encoded education level. `int64` | 0.00% | Fairness/context analysis only. | No for automated recovery/action decisions pending review. | Not a label. | Sensitive/proxy discrimination risk; includes undocumented/other codes. |
| `MARRIAGE` | Encoded marital status. `int64` | 0.00% | Fairness/context analysis only. | No for automated recovery/action decisions pending review. | Not a label. | Sensitive/proxy discrimination risk; includes code 0 in the data. |
| `AGE` | Client age in years. `int64` | 0.00% | Cohort description and fairness auditing only. | No for automated recovery/action decisions pending review. | Not a label. | Sensitive/proxy discrimination and policy risk. |
| `PAY_0` | Repayment status for September 2005. `int64` | 0.00% | Strong delinquency-history signal for research. | Conditionally, only if known before the target outcome and permitted by governance. | Not a label. | High if the target/decision timing is not strictly after this status. |
| `PAY_2` | Repayment status for August 2005. `int64` | 0.00% | Delinquency-history research. | Conditionally, with time alignment and governance. | Not a label. | Future-within-window leakage if used for an earlier decision. |
| `PAY_3` | Repayment status for July 2005. `int64` | 0.00% | Delinquency-history research. | Conditionally, with time alignment and governance. | Not a label. | Same time-order risk as `PAY_2`. |
| `PAY_4` | Repayment status for June 2005. `int64` | 0.00% | Delinquency-history research. | Conditionally, with time alignment and governance. | Not a label. | Same time-order risk as `PAY_2`. |
| `PAY_5` | Repayment status for May 2005. `int64` | 0.00% | Delinquency-history research. | Conditionally, with time alignment and governance. | Not a label. | Same time-order risk as `PAY_2`. |
| `PAY_6` | Repayment status for April 2005. `int64` | 0.00% | Delinquency-history research. | Conditionally, with time alignment and governance. | Not a label. | Same time-order risk as `PAY_2`. |
| `BILL_AMT1` | September 2005 bill statement amount, NT dollars. `int64` | 0.00% | Balance-pattern exploration. | Conditionally, only after confirming availability at cutoff. | Not a label. | High if the amount is posted after the future decision time. |
| `BILL_AMT2` | August 2005 bill statement amount, NT dollars. `int64` | 0.00% | Balance-pattern exploration. | Conditionally, with time alignment. | Not a label. | Same time-order risk as `BILL_AMT1`. |
| `BILL_AMT3` | July 2005 bill statement amount, NT dollars. `int64` | 0.00% | Balance-pattern exploration. | Conditionally, with time alignment. | Not a label. | Same time-order risk as `BILL_AMT1`. |
| `BILL_AMT4` | June 2005 bill statement amount, NT dollars. `int64` | 0.00% | Balance-pattern exploration. | Conditionally, with time alignment. | Not a label. | Same time-order risk as `BILL_AMT1`. |
| `BILL_AMT5` | May 2005 bill statement amount, NT dollars. `int64` | 0.00% | Balance-pattern exploration. | Conditionally, with time alignment. | Not a label. | Same time-order risk as `BILL_AMT1`. |
| `BILL_AMT6` | April 2005 bill statement amount, NT dollars. `int64` | 0.00% | Balance-pattern exploration. | Conditionally, with time alignment. | Not a label. | Same time-order risk as `BILL_AMT1`. |
| `PAY_AMT1` | Previous payment amount in September 2005, NT dollars. `int64` | 0.00% | Payment-behavior exploration. | Conditionally, only if posted before cutoff. | Not a label. | High if a payment outcome is included in the feature window after decision time. |
| `PAY_AMT2` | Previous payment amount in August 2005, NT dollars. `int64` | 0.00% | Payment-behavior exploration. | Conditionally, with time alignment. | Not a label. | Same time-order risk as `PAY_AMT1`. |
| `PAY_AMT3` | Previous payment amount in July 2005, NT dollars. `int64` | 0.00% | Payment-behavior exploration. | Conditionally, with time alignment. | Not a label. | Same time-order risk as `PAY_AMT1`. |
| `PAY_AMT4` | Previous payment amount in June 2005, NT dollars. `int64` | 0.00% | Payment-behavior exploration. | Conditionally, with time alignment. | Not a label. | Same time-order risk as `PAY_AMT1`. |
| `PAY_AMT5` | Previous payment amount in May 2005, NT dollars. `int64` | 0.00% | Payment-behavior exploration. | Conditionally, with time alignment. | Not a label. | Same time-order risk as `PAY_AMT1`. |
| `PAY_AMT6` | Previous payment amount in April 2005, NT dollars. `int64` | 0.00% | Payment-behavior exploration. | Conditionally, with time alignment. | Not a label. | Same time-order risk as `PAY_AMT1`. |
| `default payment next month` | Binary default-payment response (0/1). `int64` | 0.00% | Useful only to study credit-default prediction concepts. | No; it is the target in this source. | **Outcome label for default payment**, not for revenue recovery. | Direct leakage if included as a feature; it must never be renamed or treated as recovery success. |

**Important distributions:** `LIMIT_BAL` median is 140,000 NT dollars (range 10,000–1,000,000). Ages range from 21 to 79 (median 34). The payment-status columns include values from -2 to 8; their documented meaning must be retained rather than treated as an unconstrained continuous scale.

**Decision:** Retain for separate exploratory credit-risk and temporal-feature analysis only. Do not use it to claim recovery effectiveness or train a recovery-action model; it lacks recovery outcomes and is geographically, temporally, and domain-specific.

## Cross-dataset conclusion

The two datasets must remain separate. Online Retail is UK retail transaction-line data; Default of Credit Card Clients is client-level Taiwan credit-risk data. Their unit of analysis, time coverage, geography, variables, and outcomes differ. There is no scientific basis in the source data for joining them, and no `recovery_success`, `recovery_action`, or `recovered_amount` field has been created.
