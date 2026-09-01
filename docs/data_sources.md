# RecoverAI Data Sources

This investigation uses two separate public UCI Machine Learning Repository datasets. Their original archives are downloaded locally to `ml/data/external/`, remain unchanged, and are intentionally excluded from the public Git repository. This document retains the official source URLs and licenses so the datasets can be obtained reproducibly without committing their files. They are not merged, transformed, or used to create recovery labels.

## UCI Online Retail

- **Official source:** UCI Machine Learning Repository
- **URL:** <https://archive.ics.uci.edu/dataset/352/online+retail>
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Records / columns:** 541,909 transaction-line records; 8 columns in the workbook (UCI describes 6 features plus identifier fields)
- **Original filename:** `Online Retail.xlsx`, retained inside `ml/data/external/online+retail.zip`
- **What it represents:** Transactions from a UK-based, registered non-store online retailer from 1 December 2010 through 9 December 2011. The business mainly sells gifts and many customers are wholesalers.
- **Relevant fields:** `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, and `Country`. An invoice beginning with `C` denotes a cancellation.
- **Potential RecoverAI uses:** exploratory analysis of transaction timing, cancellation patterns, product/customer concentration, purchase values, and customer-history aggregation design.
- **Fields not to use directly:** `CustomerID` and `InvoiceNo` are identifiers, not generalizable model features; `Description` may be high-cardinality/free text. Future-derived customer aggregates must be time-bounded to avoid leakage.
- **Limitations:** No payment-failure, collection, recovery-action, recovered-amount, chargeback, or contact-outcome label exists. It is retail transaction data, not a revenue-recovery dataset.
- **Data description:** The source describes it as transactional data from a UK online retailer; it is not described as synthetic by UCI.
- **Contribution to RecoverAI:** Useful for understanding pre-event retail behavior and cancellation signals, but cannot validate recovery decisions or recovery outcomes on its own.

## UCI Default of Credit Card Clients

- **Official source:** UCI Machine Learning Repository
- **URL:** <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Records / columns:** 30,000 client records; 25 columns in the workbook (23 explanatory variables, an ID, and one response label)
- **Original filename:** `default of credit card clients.xls`, retained inside `ml/data/external/default+of+credit+card+clients.zip`
- **What it represents:** A Taiwan credit-card-client study of default payment prediction. The response is whether the client defaulted in the next month.
- **Relevant fields:** `LIMIT_BAL`, `AGE`, payment-status history (`PAY_0`, `PAY_2`–`PAY_6`), bill amounts (`BILL_AMT1`–`BILL_AMT6`), previous-payment amounts (`PAY_AMT1`–`PAY_AMT6`), and the response `default payment next month`.
- **Potential RecoverAI uses:** exploratory study of delinquency histories, payment behavior, balance patterns, feature time alignment, and default-risk limitations.
- **Fields not to use directly:** `ID` is only an identifier. `SEX`, `EDUCATION`, `MARRIAGE`, and `AGE` are sensitive or demographic attributes and should not be used for automated action decisions without a future legal, ethical, and fairness review.
- **Limitations:** The binary default label is not a recovery label and does not say whether a recovery action succeeded. The data is a historical Taiwan credit-card study and is not necessarily representative of Razorpay merchants, customers, or current payment behavior.
- **Data description:** UCI describes it as a research dataset about Taiwan credit-card clients; it is not described as synthetic by UCI.
- **Contribution to RecoverAI:** Useful for learning how to analyze credit-risk inputs separately from recovery outcomes, but insufficient to train or evaluate a recovery system.
