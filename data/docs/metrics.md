# Evaluation Metrics for Fraud Detection

## Classification Metrics

### Precision
**Formula**: `TP / (TP + FP)`

The fraction of predicted fraud cases that are actually fraud. High precision means few false alarms.

**In PaySim context**: If a model flags 100 transactions as fraud and 90 are actually fraud, precision = 0.90. Low precision means investigators waste time on legitimate transactions.

### Recall (Sensitivity / True Positive Rate)
**Formula**: `TP / (TP + FN)`

The fraction of actual fraud cases that are correctly detected. High recall means few missed frauds.

**In PaySim context**: If there are 8,213 actual fraud cases and the model catches 7,000, recall = 0.853. Low recall means real fraud goes undetected — **this is costly**.

### F1 Score
**Formula**: `2 * (Precision * Recall) / (Precision + Recall)`

Harmonic mean of precision and recall. Useful when you need a single balanced metric.

### F-beta Score
**Formula**: `(1 + B^2) * (Precision * Recall) / (B^2 * Precision + Recall)`

- **B = 2** (F2 Score): Weights recall 2x more than precision. Preferred for fraud detection because **missing fraud is more expensive than a false alarm**.
- **B = 0.5** (F0.5 Score): Weights precision more. Use when false positives are very costly (e.g., freezing legitimate accounts).

### False Positive Rate (FPR)
**Formula**: `FP / (FP + TN)`

The fraction of legitimate transactions incorrectly flagged as fraud. Must be kept very low because legitimate transactions vastly outnumber fraud (99.87% vs 0.13%).

**Target**: FPR < 1% — even 1% FPR on 6.35M legitimate transactions = 63,500 false alarms.

---

## Cost-Sensitive Metrics

### Cost of Missed Fraud (False Negative Cost)
Each missed fraud transaction has a direct financial cost equal to the transaction amount. In PaySim:
- Mean fraud transaction amount: significantly higher than overall mean
- A single missed fraud in TRANSFER can cost hundreds of thousands

```python
missed_fraud_cost = df[(df['isFraud'] == 1) & (df['predicted'] == 0)]['amount'].sum()
```

### Cost of False Alarm (False Positive Cost)
Each false alarm costs investigation time (analyst hours) but no direct financial loss. Estimated at ~$50-$100 per case in real systems.

```python
false_alarm_cost = ((df['isFraud'] == 0) & (df['predicted'] == 1)).sum() * 50  # $50 per investigation
```

### Net Savings
**Formula**: `fraud_caught_value - false_alarm_cost - missed_fraud_cost`

```python
caught_value = df[(df['isFraud'] == 1) & (df['predicted'] == 1)]['amount'].sum()
missed_cost = df[(df['isFraud'] == 1) & (df['predicted'] == 0)]['amount'].sum()
fa_cost = ((df['isFraud'] == 0) & (df['predicted'] == 1)).sum() * 50
net_savings = caught_value - fa_cost - missed_cost
```

---

## Threshold-Independent Metrics

### AUROC (Area Under ROC Curve)
Measures the model's ability to distinguish fraud from non-fraud across all thresholds. Range: 0.5 (random) to 1.0 (perfect).

### AUPRC (Area Under Precision-Recall Curve)
More informative than AUROC for **highly imbalanced** datasets like PaySim (0.13% fraud). AUROC can be misleadingly high even for poor models when negatives dominate.

**Recommendation**: Always report AUPRC alongside AUROC for PaySim.

---

## Class Imbalance Considerations

PaySim has a **775:1** imbalance ratio. This affects metrics:

| Metric | Impact of Imbalance |
|---|---|
| Accuracy | Misleading — 99.87% accuracy by predicting all non-fraud |
| Precision | Drops quickly with even a few false positives |
| Recall | Unaffected by imbalance, depends only on fraud detection |
| AUROC | Can be artificially high |
| AUPRC | Accurately reflects performance on minority class |

**Never use accuracy** as a primary metric for PaySim fraud detection.

---

## Pandas Code for Metrics

```python
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, roc_auc_score, average_precision_score

y_true = df['isFraud']
y_pred = df['predicted']  # binary predictions
y_prob = df['fraud_probability']  # probability scores (if available)

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
f2 = fbeta_score(y_true, y_pred, beta=2)
auroc = roc_auc_score(y_true, y_prob)
auprc = average_precision_score(y_true, y_prob)
```

---

## Baseline Performance

### isFlaggedFraud as a baseline detector:
- **Precision**: 16/16 = 1.0 (all flagged transactions are fraud)
- **Recall**: 16/8213 = 0.0019 (catches only 0.19% of fraud)
- **F1**: 0.0039
- **Verdict**: Useless as a detector despite perfect precision — misses 99.8% of fraud

### "Flag all TRANSFER + CASH_OUT" baseline:
- **Recall**: 1.0 (catches all fraud by definition)
- **Precision**: 8213 / 2,770,409 = 0.003 (99.7% false positive rate)
- **Verdict**: Too many false positives to be practical
