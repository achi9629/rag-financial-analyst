# PaySim Dataset Schema

## Overview

PaySim is a synthetic dataset simulating mobile money transactions based on a real dataset from a mobile money service in an African country. It contains **6,362,620 transactions** across **11 columns** spanning 743 simulated hours (~30 days).

## Columns

| Column | Type | Description |
|---|---|---|
| `step` | int64 | Time unit of the simulation. **1 step = 1 hour**. Range: 1–743 (approximately 30 days). |
| `type` | object (str) | Transaction type. One of: `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`. |
| `amount` | float64 | Transaction amount in local currency. Range: 0 – 92,445,516.64. Mean: ~179,862. |
| `nameOrig` | object (str) | Customer ID of the sender/originator. Format: `C` followed by digits (e.g., `C1231006815`). |
| `oldbalanceOrg` | float64 | Sender's balance before the transaction. Range: 0 – 59,585,040.37. |
| `newbalanceOrig` | float64 | Sender's balance after the transaction. Range: 0 – 49,585,040.37. |
| `nameDest` | object (str) | ID of the recipient. Format: `C` + digits for customers, `M` + digits for merchants. |
| `oldbalanceDest` | float64 | Recipient's balance before the transaction. Range: 0 – 356,015,889.35. |
| `newbalanceDest` | float64 | Recipient's balance after the transaction. Range: 0 – 356,179,278.92. |
| `isFraud` | int64 | **Ground truth** fraud label. 1 = fraudulent, 0 = legitimate. |
| `isFlaggedFraud` | int64 | Rule-based fraud flag from the legacy system. 1 = flagged, 0 = not flagged. |

## Transaction Type Distribution

| Type | Count | Percentage |
|---|---|---|
| CASH_OUT | 2,237,500 | 35.2% |
| PAYMENT | 2,151,495 | 33.8% |
| CASH_IN | 1,399,284 | 22.0% |
| TRANSFER | 532,909 | 8.4% |
| DEBIT | 41,432 | 0.7% |

## Fraud Statistics

- **Total fraudulent transactions**: 8,213 (0.129% of all transactions)
- **Class imbalance ratio**: ~775:1 (legitimate : fraud)
- Fraud **only** occurs in two transaction types:
  - `CASH_OUT`: 4,116 fraudulent (0.184% of all CASH_OUT)
  - `TRANSFER`: 4,097 fraudulent (0.769% of all TRANSFER)
- `CASH_IN`, `PAYMENT`, and `DEBIT` have **zero** fraud cases.

## isFlaggedFraud Details

- Only **16 transactions** are flagged by the legacy rule-based system.
- All 16 are `TRANSFER` type.
- Minimum flagged amount: >350,000 (the legacy system only flags very large transfers).
- This means the legacy flag catches <0.2% of actual fraud — it is nearly useless.

## Data Quality

- **No null values** across any column.
- No duplicate rows by design (synthetic simulation).
- Balance columns can be 0.0 (legitimate for new or emptied accounts).

## Key Relationships

- For non-fraud transactions: `newbalanceOrig ≈ oldbalanceOrg - amount` (for outgoing types).
- For fraud transactions: balance fields are often **inconsistent** — this is a key fraud signal.
- `nameDest` starting with `M` = merchant (merchants cannot be senders, only receivers in PAYMENT).
- `step` is monotonically non-decreasing; multiple transactions can share the same step (same hour).

## Descriptive Statistics

| Statistic | step | amount | oldbalanceOrg | newbalanceOrig | oldbalanceDest | newbalanceDest |
|---|---|---|---|---|---|---|
| mean | 243 | 179,862 | 833,883 | 855,114 | 1,100,702 | 1,224,996 |
| std | 142 | 603,858 | 2,888,243 | 2,924,049 | 3,399,180 | 3,674,129 |
| min | 1 | 0 | 0 | 0 | 0 | 0 |
| 25% | 156 | 13,390 | 0 | 0 | 0 | 0 |
| 50% | 239 | 74,872 | 14,208 | 0 | 132,706 | 214,661 |
| 75% | 335 | 208,722 | 107,315 | 144,258 | 943,037 | 1,111,909 |
| max | 743 | 92,445,517 | 59,585,040 | 49,585,040 | 356,015,889 | 356,179,279 |
