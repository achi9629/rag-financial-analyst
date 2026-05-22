# Fraud Patterns — Detection Logic for PaySim

## Pattern 1: Smurfing (Structuring)

**Definition**: Breaking a large transaction into multiple smaller transactions to avoid reporting thresholds (e.g., 10 lakh / $10,000).

**Detection logic**:
```python
# Find accounts making multiple transactions just below the reporting threshold
threshold = 1000000  # 10 lakh
window_hours = 24    # within 24 hours

# Transactions just below the threshold (within 10%)
near_threshold = df[(df['amount'] >= threshold * 0.9) & (df['amount'] < threshold)]

# Group by sender within time window
structuring = df[df['amount'] < threshold].groupby('nameOrig').agg(
    txn_count=('amount', 'size'),
    total_amount=('amount', 'sum'),
    time_span=('step', lambda x: x.max() - x.min())
).reset_index()

# Flag: total exceeds threshold but individual transactions don't
structuring_suspects = structuring[
    (structuring['total_amount'] >= threshold) &
    (structuring['txn_count'] >= 3) &
    (structuring['time_span'] <= window_hours)
]
```

**Indicators**:
- Multiple transactions from the same sender within a short period
- Each transaction is below the reporting threshold
- Sum of transactions exceeds the threshold
- Transactions are often round numbers or near-identical amounts

---

## Pattern 2: Layering

**Definition**: Moving money through a chain of accounts to obscure its origin. Funds flow: Account A -> B -> C -> D, making it hard to trace back to A.

**Detection logic**:
```python
# Find chains: A sends to B, B sends to C (within a time window)
transfers = df[df['type'] == 'TRANSFER'][['step', 'nameOrig', 'nameDest', 'amount']].copy()

# Self-join: find B who receives from A and sends to C
layer1 = transfers.rename(columns={'nameOrig': 'A', 'nameDest': 'B', 'step': 'step_1', 'amount': 'amount_1'})
layer2 = transfers.rename(columns={'nameOrig': 'B_sender', 'nameDest': 'C', 'step': 'step_2', 'amount': 'amount_2'})

chains = layer1.merge(layer2, left_on='B', right_on='B_sender')
chains = chains[
    (chains['step_2'] > chains['step_1']) &           # B sends after receiving
    (chains['step_2'] - chains['step_1'] <= 24) &     # within 24 hours
    (chains['amount_2'] >= chains['amount_1'] * 0.8)   # similar amount (minus fee)
]
```

**Indicators**:
- Funds pass through 3+ accounts in sequence
- Time between hops is short (< 24 hours)
- Amounts are similar across hops (minus small fees)
- Intermediate accounts have little other activity

---

## Pattern 3: Mule Accounts

**Definition**: Accounts used as intermediaries to receive and forward fraudulent funds. Mules receive money from victims and quickly forward it onward (often via CASH_OUT).

**Detection logic**:
```python
# Mule indicators: receives TRANSFER then does CASH_OUT quickly
received = df[df['type'] == 'TRANSFER'][['step', 'nameDest', 'amount']].rename(
    columns={'nameDest': 'account', 'step': 'recv_step', 'amount': 'recv_amount'}
)
sent = df[df['type'] == 'CASH_OUT'][['step', 'nameOrig', 'amount']].rename(
    columns={'nameOrig': 'account', 'step': 'send_step', 'amount': 'send_amount'}
)

mule_candidates = received.merge(sent, on='account')
mule_candidates = mule_candidates[
    (mule_candidates['send_step'] - mule_candidates['recv_step'] <= 2) &  # within 2 hours
    (mule_candidates['send_amount'] >= mule_candidates['recv_amount'] * 0.9)  # similar amount
]

mule_accounts = mule_candidates['account'].unique()
```

**Indicators**:
- Receives TRANSFER and immediately does CASH_OUT (within 1-2 hours)
- Forwarded amount is nearly equal to received amount
- Account has very few other transactions
- Account was recently created (low step number for first transaction)

---

## Pattern 4: Round-Trip Transfers

**Definition**: Money is sent from A to B and then back from B to A. Used to simulate legitimate activity, inflate transaction volumes, or test compromised accounts.

**Detection logic**:
```python
transfers = df[df['type'] == 'TRANSFER'][['step', 'nameOrig', 'nameDest', 'amount']]

fwd = transfers.rename(columns={'nameOrig': 'A', 'nameDest': 'B', 'step': 'step_fwd', 'amount': 'amount_fwd'})
rev = transfers.rename(columns={'nameOrig': 'B_sender', 'nameDest': 'A_recv', 'step': 'step_rev', 'amount': 'amount_rev'})

roundtrip = fwd.merge(rev, left_on=['A', 'B'], right_on=['A_recv', 'B_sender'])
roundtrip = roundtrip[
    (roundtrip['step_rev'] > roundtrip['step_fwd']) &
    (roundtrip['step_rev'] - roundtrip['step_fwd'] <= 48) &  # within 48 hours
    (abs(roundtrip['amount_fwd'] - roundtrip['amount_rev']) / roundtrip['amount_fwd'] < 0.1)  # within 10%
]
```

**Indicators**:
- Bidirectional transfers between the same pair
- Similar amounts in both directions
- Short time gap between forward and return transfer

---

## Pattern 5: Account Draining

**Definition**: A single transaction that empties the sender's entire balance. Common in account takeover fraud.

**Detection logic**:
```python
draining = df[
    (df['type'].isin(['TRANSFER', 'CASH_OUT'])) &
    (df['oldbalanceOrg'] > 0) &
    (df['newbalanceOrig'] == 0) &
    (df['amount'] == df['oldbalanceOrg'])
]
```

**Indicators**:
- Transaction amount equals the sender's full balance
- Sender balance goes to exactly 0
- Transaction type is TRANSFER or CASH_OUT
- Often the sender's only transaction (first and last)

---

## Pattern 6: Rapid Successive Transfers

**Definition**: An account makes multiple outgoing transfers in quick succession, distributing funds to multiple recipients.

**Detection logic**:
```python
rapid = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].groupby('nameOrig').agg(
    txn_count=('amount', 'size'),
    unique_recipients=('nameDest', 'nunique'),
    total_sent=('amount', 'sum'),
    time_span=('step', lambda x: x.max() - x.min()),
    first_step=('step', 'min'),
    last_step=('step', 'max')
).reset_index()

fan_out = rapid[
    (rapid['txn_count'] >= 5) &
    (rapid['unique_recipients'] >= 3) &
    (rapid['time_span'] <= 6)  # within 6 hours
]
```

**Indicators**:
- 5+ transactions from one sender within a few hours
- Multiple distinct recipients
- Suggests distributing stolen funds to multiple mule accounts

---

## Pattern Summary Table

| Pattern | Key Signal | Transaction Type | Time Window |
|---|---|---|---|
| Smurfing | Multiple txns below threshold, sum exceeds it | Any | 24 hours |
| Layering | A->B->C chain with similar amounts | TRANSFER | 24 hours between hops |
| Mule Account | Receive TRANSFER, immediate CASH_OUT | TRANSFER + CASH_OUT | 1-2 hours |
| Round-Trip | A->B then B->A with similar amount | TRANSFER | 48 hours |
| Account Draining | amount == oldbalanceOrg, newbalanceOrig == 0 | TRANSFER, CASH_OUT | Single transaction |
| Rapid Fan-Out | 5+ txns to multiple recipients in short window | TRANSFER, CASH_OUT | 6 hours |
