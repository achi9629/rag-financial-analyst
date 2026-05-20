# Domain Glossary — PaySim & Financial Fraud

## Transaction Types

### CASH_IN
Money deposited into an account from cash. The recipient's balance increases.
- Sender: external cash source
- Receiver: customer account (`C` prefix)
- Fraud: **never** (no fraud cases in CASH_IN)
- Count in dataset: 1,399,284

### CASH_OUT
Money withdrawn from an account as cash. The sender's balance decreases.
- Sender: customer account (`C` prefix)
- Receiver: merchant or agent (`M` prefix)
- Fraud: **yes** — 4,116 fraudulent CASH_OUT transactions (0.184% of all CASH_OUT)
- Count in dataset: 2,237,500
- Fraud pattern: attacker takes over account then drains balance via CASH_OUT

### PAYMENT
Purchase of goods or services from a merchant. Sender pays, merchant receives.
- Sender: customer account (`C` prefix)
- Receiver: merchant (`M` prefix)
- Fraud: **never** (no fraud cases in PAYMENT)
- Count in dataset: 2,151,495

### TRANSFER
Person-to-person money transfer between two customer accounts.
- Sender: customer account (`C` prefix)
- Receiver: customer account (`C` prefix)
- Fraud: **yes** — 4,097 fraudulent TRANSFER transactions (0.769% of all TRANSFER)
- Count in dataset: 532,909
- Fraud pattern: attacker transfers victim's funds to a mule account, then CASH_OUT

### DEBIT
Direct debit — automated withdrawal from a customer account (e.g., bill payment).
- Sender: customer account (`C` prefix)
- Receiver: system/merchant
- Fraud: **never** (no fraud cases in DEBIT)
- Count in dataset: 41,432

---

## Account ID Prefixes

| Prefix | Meaning | Can Send? | Can Receive? |
|---|---|---|---|
| `C` | Customer account | Yes (all types) | Yes (CASH_IN, TRANSFER) |
| `M` | Merchant account | No | Yes (PAYMENT, CASH_OUT) |

---

## Label Columns

### isFraud
Ground truth label indicating whether a transaction is fraudulent.
- `0` = legitimate transaction
- `1` = fraudulent transaction
- Total fraud: 8,213 out of 6,362,620 (0.129%)
- Only appears in TRANSFER and CASH_OUT types
- This is the **authoritative** label for evaluation

### isFlaggedFraud
Legacy rule-based fraud detection flag from the simulation's internal system.
- `0` = not flagged
- `1` = flagged as suspicious
- Only **16 transactions** are flagged (all TRANSFER type)
- Minimum flagged amount: > 350,000
- **Extremely poor recall**: catches <0.2% of actual fraud
- This flag should NOT be used as a reliable fraud indicator — it exists to demonstrate the weakness of simple rule-based systems

---

## Time Column

### step
Simulated time unit where **1 step = 1 hour**.
- Range: 1 to 743 (approximately 30.96 days or ~1 month)
- Multiple transactions can occur in the same step
- To convert to days: `day = (step - 1) // 24 + 1`
- To get hour of day: `hour = (step - 1) % 24`

---

## Balance Columns

### oldbalanceOrg / newbalanceOrig
Sender's account balance **before** and **after** the transaction.
- For legitimate outgoing transactions: `newbalanceOrig ≈ oldbalanceOrg - amount`
- Discrepancy between expected and actual balance is a fraud signal

### oldbalanceDest / newbalanceDest
Receiver's account balance **before** and **after** the transaction.
- For legitimate incoming transactions: `newbalanceDest ≈ oldbalanceDest + amount`
- Note: merchant (`M` prefix) balances are not tracked — they show as 0.0

---

## Key Financial Terms

### KYC (Know Your Customer)
Regulatory requirement to verify customer identity before allowing financial services. Relevant to mule account detection — mule accounts often bypass or use fake KYC.

### AML (Anti-Money Laundering)
Laws and procedures to prevent criminals from disguising illegally obtained funds as legitimate income. PaySim patterns like structuring and layering are AML concerns.

### CTR (Currency Transaction Report)
Mandatory report filed for transactions above a threshold (e.g., 10 lakh in India, $10,000 in the US). Used to detect large-value money laundering.

### SAR (Suspicious Activity Report)
Report filed when a financial institution suspects a transaction may involve money laundering or fraud, regardless of amount.

### PEP (Politically Exposed Person)
An individual in a prominent public position. PEP transactions require enhanced due diligence under AML regulations.
