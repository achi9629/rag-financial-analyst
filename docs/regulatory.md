# Regulatory Compliance Rules for Financial Transactions

## Overview

These rules encode regulatory reporting thresholds and compliance requirements relevant to fraud detection in mobile money systems. The LLM should reference these when generating code for compliance-related queries.

---

## Currency Transaction Reporting (CTR)

### India — RBI Guidelines
- **Threshold**: Transactions >= 10,00,000 (10 lakh / 1,000,000) must be reported
- **Applies to**: All transaction types (cash and non-cash)
- **Reporting entity**: Banks and financial institutions must file CTR with Financial Intelligence Unit (FIU-IND)
- **Timeline**: Within 15 days of the transaction

**Detection code**:
```python
reportable = df[df['amount'] >= 1000000]
```

### United States — FinCEN
- **Threshold**: Transactions > $10,000 must be reported via CTR
- **Applies to**: Cash transactions
- **Reporting entity**: Banks file with FinCEN (Financial Crimes Enforcement Network)

### European Union — AMLD
- **Threshold**: Transactions >= EUR 15,000 require enhanced due diligence
- **Applies to**: Occasional transactions (non-account holders)

---

## Structuring (Smurfing) Detection

### Definition
Deliberately breaking a transaction into smaller amounts to avoid the CTR reporting threshold. This is illegal in most jurisdictions regardless of whether the underlying funds are legitimate.

### India — PMLA (Prevention of Money Laundering Act)
- Structuring to avoid the 10 lakh threshold is a criminal offense under PMLA 2002
- Even if individual transactions are legitimate, the act of splitting to evade reporting is itself illegal

### Detection Rules

**Rule S1 — Near-Threshold Clustering**:
```python
threshold = 1000000  # 10 lakh
# Transactions between 90% and 100% of threshold
near_threshold = df[(df['amount'] >= threshold * 0.9) & (df['amount'] < threshold)]
# Multiple near-threshold transactions from same sender
suspects = near_threshold.groupby('nameOrig').filter(lambda x: len(x) >= 2)
```

**Rule S2 — Aggregate Exceeds Threshold**:
```python
# Same sender, total exceeds threshold within 24 hours, no single txn above it
daily_agg = df.groupby('nameOrig').agg(
    total=('amount', 'sum'),
    max_single=('amount', 'max'),
    count=('amount', 'size'),
    time_span=('step', lambda x: x.max() - x.min())
).reset_index()

structuring = daily_agg[
    (daily_agg['total'] >= threshold) &
    (daily_agg['max_single'] < threshold) &
    (daily_agg['count'] >= 3) &
    (daily_agg['time_span'] <= 24)
]
```

**Rule S3 — Round Number Pattern**:
```python
# Transactions in round amounts just below threshold
round_amounts = df[
    (df['amount'] % 10000 == 0) &
    (df['amount'] >= 500000) &
    (df['amount'] < 1000000)
]
```

---

## Suspicious Activity Reporting (SAR)

### When to File
A SAR must be filed when a financial institution suspects a transaction involves:
1. Funds from illegal activity
2. Attempts to evade reporting requirements (structuring)
3. No business or lawful purpose
4. Unusual patterns inconsistent with customer profile

### SAR Triggers in PaySim Context

**Trigger 1 — Unusual transaction pattern**:
```python
# Sender's transaction is 3x their historical average
sender_avg = df.groupby('nameOrig')['amount'].transform('mean')
unusual = df[df['amount'] > 3 * sender_avg]
```

**Trigger 2 — Rapid movement of funds**:
```python
# Funds received and immediately forwarded (within 1-2 hours)
received = df[df['type'] == 'TRANSFER'][['step', 'nameDest', 'amount']].rename(
    columns={'nameDest': 'account', 'step': 'recv_step'}
)
forwarded = df[df['type'].isin(['CASH_OUT', 'TRANSFER'])][['step', 'nameOrig', 'amount']].rename(
    columns={'nameOrig': 'account', 'step': 'fwd_step'}
)
rapid = received.merge(forwarded, on='account')
rapid = rapid[(rapid['fwd_step'] - rapid['recv_step'] >= 0) & (rapid['fwd_step'] - rapid['recv_step'] <= 2)]
```

**Trigger 3 — Account used as pass-through**:
```python
# Account receives and sends similar amounts with minimal balance retention
account_flow = df.groupby('nameOrig').agg(
    total_sent=('amount', 'sum'),
    txn_count=('amount', 'size')
).reset_index()

account_received = df.groupby('nameDest').agg(
    total_received=('amount', 'sum')
).reset_index().rename(columns={'nameDest': 'nameOrig'})

flow = account_flow.merge(account_received, on='nameOrig', how='inner')
# Pass-through: sends out >= 90% of what it receives
pass_through = flow[flow['total_sent'] >= flow['total_received'] * 0.9]
```

---

## KYC (Know Your Customer) Requirements

### Tiered KYC for Mobile Money
| Tier | ID Required | Transaction Limit | Balance Limit |
|---|---|---|---|
| Minimum KYC | Phone number only | 10,000/month | 10,000 |
| Small KYC | Basic ID (Aadhaar) | 100,000/month | 100,000 |
| Full KYC | Full ID + address proof | No limit | No limit |

### KYC Red Flags in PaySim
- Account transacting above expected tier limits
- New account with immediate high-value transactions
- Account with no prior history making large transfers

**Detection code**:
```python
# First transaction by each account
first_txn = df.sort_values('step').groupby('nameOrig').first().reset_index()
# New accounts with high-value first transactions
suspicious_new = first_txn[first_txn['amount'] > 100000]
```

---

## AML (Anti-Money Laundering) Compliance

### Three Stages of Money Laundering
1. **Placement**: Introducing illicit funds into the financial system
   - PaySim signal: Large CASH_IN transactions, especially to new accounts
2. **Layering**: Moving funds through multiple accounts/transactions to obscure origin
   - PaySim signal: Chain of TRANSFER transactions (A->B->C->D) with similar amounts
3. **Integration**: Withdrawing cleaned funds for legitimate use
   - PaySim signal: CASH_OUT after a series of transfers

### AML Detection Pipeline
```python
# Stage 1: Large placements
placements = df[(df['type'] == 'CASH_IN') & (df['amount'] > 500000)]

# Stage 2: Layering chains (see patterns.md for full implementation)
transfers = df[df['type'] == 'TRANSFER']
layer1 = transfers.rename(columns={'nameOrig': 'A', 'nameDest': 'B', 'step': 's1', 'amount': 'a1'})
layer2 = transfers.rename(columns={'nameOrig': 'B2', 'nameDest': 'C', 'step': 's2', 'amount': 'a2'})
chains = layer1.merge(layer2, left_on='B', right_on='B2')
chains = chains[(chains['s2'] > chains['s1']) & (chains['s2'] - chains['s1'] <= 24)]

# Stage 3: Integration via CASH_OUT
cashouts = df[(df['type'] == 'CASH_OUT') & (df['amount'] > 200000)]
```

---

## Reporting Thresholds Summary

| Jurisdiction | Threshold | Currency | Report Type |
|---|---|---|---|
| India (RBI) | 10,00,000 | INR | CTR to FIU-IND |
| USA (FinCEN) | 10,000 | USD | CTR |
| EU (AMLD) | 15,000 | EUR | Enhanced Due Diligence |
| UK (NCA) | Any suspicious | GBP | SAR to NCA |

---

## PaySim-Specific Regulatory Notes

- PaySim amounts are in a generic currency unit (not explicitly INR/USD)
- For this project, we use **1,000,000** as the regulatory threshold (analogous to 10 lakh INR)
- The `isFlaggedFraud` column represents a naive rule-based system that only flags TRANSFER > 350,000 — this is far above/below most regulatory thresholds and catches almost nothing
- A proper compliance system would flag based on the rules above, not just single transaction size
