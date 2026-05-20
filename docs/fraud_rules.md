# Fraud Detection Rules for PaySim

## Overview

These rules define how to detect fraudulent transactions in the PaySim dataset. The LLM should use these rules when generating pandas code for fraud-related queries. All rules are derived from known fraud patterns in mobile money systems.

---

## Rule 1: Balance Mismatch Detection

**Signal**: The sender's balance change does not match the transaction amount.

**Formula**:
```python
balance_diff = df['oldbalanceOrg'] - df['newbalanceOrig']
mismatch = (balance_diff - df['amount']).abs() > 1.0  # tolerance for float precision
```

**Why it matters**: In legitimate transactions, `oldbalanceOrg - amount ≈ newbalanceOrig`. Fraudulent transactions often manipulate balances — the sender's account may show no deduction, or the deduction doesn't match the amount.

**In PaySim**: Many fraud cases have `oldbalanceOrg == amount` and `newbalanceOrig == 0` (account fully drained), but `oldbalanceDest == 0` and `newbalanceDest == 0` (recipient balance unchanged despite receiving funds). This zero-destination pattern is a strong fraud indicator.

---

## Rule 2: Destination Balance Anomaly

**Signal**: Recipient receives funds but their balance doesn't increase.

**Formula**:
```python
dest_mismatch = (df['newbalanceDest'] == df['oldbalanceDest']) & (df['amount'] > 0)
```

**Why it matters**: If money is sent but the destination balance is unchanged, the transaction may be fraudulent (funds diverted or account is a shell).

---

## Rule 3: Account Draining (Zero-Out)

**Signal**: A transaction completely drains the sender's balance.

**Formula**:
```python
zeroed_out = (df['oldbalanceOrg'] > 0) & (df['newbalanceOrig'] == 0) & (df['amount'] == df['oldbalanceOrg'])
```

**Why it matters**: Fraudsters typically drain the entire balance in a single transaction. Legitimate users rarely transfer their exact full balance.

**Combined with CASH_OUT or TRANSFER type**: This is an even stronger signal since fraud only occurs in these two types.

---

## Rule 4: High-Value Transaction Threshold

**Signal**: Transaction amount exceeds a threshold.

**Thresholds**:
- **Legacy system flag**: Transactions > 200,000 (but `isFlaggedFraud` only triggers on TRANSFER > 350,000 — catches only 16 out of 8,213 fraud cases)
- **Regulatory reporting**: Transactions >= 10,00,000 (10 lakh / ~$12,000 in many jurisdictions)
- **Statistical outlier**: Transactions > mean + 2*std ≈ 179,862 + 2*603,858 ≈ 1,387,578

**Formula**:
```python
high_value = df['amount'] > 200000
regulatory_threshold = df['amount'] >= 1000000
statistical_outlier = df['amount'] > df['amount'].mean() + 2 * df['amount'].std()
```

---

## Rule 5: Transaction Velocity (Frequency-Based)

**Signal**: An account makes too many transactions in a short time window.

**Formula**:
```python
velocity = df.groupby(['nameOrig', 'step']).size().reset_index(name='txn_count')
high_velocity = velocity[velocity['txn_count'] > 5]  # more than 5 txns per hour
```

**Thresholds**:
- **Suspicious**: > 3 transactions per hour from the same sender
- **High risk**: > 5 transactions per hour from the same sender
- **Critical**: > 10 transactions per hour from the same sender

**Why it matters**: Normal users don't make many transactions per hour. Rapid-fire transactions suggest automated fraud or money laundering.

---

## Rule 6: Amount Deviation from Account Norm

**Signal**: Transaction amount is significantly different from the sender's historical average.

**Formula**:
```python
sender_stats = df.groupby('nameOrig')['amount'].agg(['mean', 'std']).reset_index()
df_with_stats = df.merge(sender_stats, on='nameOrig')
df_with_stats['z_score'] = (df_with_stats['amount'] - df_with_stats['mean']) / df_with_stats['std'].clip(lower=1)
anomalous = df_with_stats['z_score'].abs() > 3
```

**Why it matters**: A sudden large transaction from an account that normally makes small payments is suspicious.

---

## Rule 7: Type-Restricted Fraud Check

**Signal**: Fraud only occurs in `TRANSFER` and `CASH_OUT` transactions.

**Formula**:
```python
fraud_eligible = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])]
```

**Why it matters**: In PaySim, `CASH_IN`, `PAYMENT`, and `DEBIT` have zero fraud cases. Any fraud detection analysis should focus on TRANSFER and CASH_OUT only.

---

## Rule 8: Sender-Receiver Pair Analysis

**Signal**: Same sender-receiver pair transacts repeatedly in a short window.

**Formula**:
```python
pair_freq = df.groupby(['nameOrig', 'nameDest']).agg(
    txn_count=('amount', 'size'),
    total_amount=('amount', 'sum'),
    time_span=('step', lambda x: x.max() - x.min())
).reset_index()
suspicious_pairs = pair_freq[(pair_freq['txn_count'] > 3) & (pair_freq['time_span'] <= 24)]
```

**Why it matters**: Repeated transfers between the same accounts in a short time suggest money laundering or mule account activity.

---

## Composite Risk Score

Combine multiple signals into a weighted risk score:

```python
risk_score = (
    0.30 * balance_mismatch +      # Rule 1
    0.20 * zeroed_out +             # Rule 3
    0.15 * high_value +             # Rule 4
    0.15 * high_velocity +          # Rule 5
    0.10 * amount_deviation +       # Rule 6
    0.10 * dest_balance_anomaly     # Rule 2
)
```

Each component is binary (0 or 1). Final score range: 0.0 (no risk) to 1.0 (maximum risk).

**Risk tiers**:
- **Low**: 0.0-0.2
- **Medium**: 0.2-0.5
- **High**: 0.5-0.8
- **Critical**: 0.8-1.0
