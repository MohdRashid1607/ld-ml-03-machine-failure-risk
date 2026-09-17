# Model Card — LD ML 03: Machine Failure Risk Detector

**Model ID:** `ld-ml-03-machine-failure-risk`  
**Version:** 0.1.0  
**Last Updated:** 2026-09-14  
**Model Type:** Unsupervised Anomaly Detection (Isolation Forest)  
**Language:** Python 3.11+  

---

> ⚠️ **CRITICAL DISCLAIMER**  
> This model **does not predict winning lottery numbers** and must never be
> used for that purpose. Anomaly scores are **not confirmed failure
> probabilities** without verified operational labels. Scores reflect
> statistical deviation from historical norms and require human review
> before operational decisions are made.

---

## 1. Model Description

The LD ML 03 model is an unsupervised anomaly detection system designed to
identify statistically abnormal operating patterns in lottery draw machines.
It uses an **Isolation Forest** as the primary detector, with a **Robust
Z-Score Detector** (median + MAD) as a comparison baseline.

The system operates exclusively on mechanical and environmental signals
(e.g. draw frequency, temperature, machine usage history). It does not
process, store, or analyse draw outcomes (winning numbers or ball positions).

### Architecture

| Component | Details |
|---|---|
| Primary Model | `sklearn.ensemble.IsolationForest` |
| Baseline Model | Robust Z-Score (median + MAD) |
| Feature Engineering | Rolling windows (30 draws), machine history, temporal stability, environment |
| Leakage Control | `shift(1)` before all rolling operations; chronological train/val/test splits |
| Persistence | `joblib` serialisation |

---

## 2. Intended Use

- **Intended users:** Lottery equipment maintenance teams and operations analysts.
- **Intended use:** Flag lottery draw machines that exhibit statistically
  unusual operating patterns for priority inspection.
- **Deployment context:** Internal operational dashboard (Hugging Face Spaces /
  Gradio) accessible to authorised staff only.

---

## 3. Out of Scope

The following uses are **explicitly out of scope and prohibited**:

- Predicting lottery draw outcomes, winning numbers, or ball sequences.
- Estimating the probability of winning for any player or ticket.
- Inferring machine state from draw results (reverse engineering of outcomes).
- Any use that could give a player an unfair advantage in a lottery.
- Replacing human engineering judgment for maintenance decisions.

---

## 4. Training Data

- **Source:** Historical lottery draw operational records (see `dataset_card.md`).
- **Training split:** All records with `draw_local_datetime` ≤ 2023-06-30.
- **Features used:** Machine history (usage counts, service intervals), distribution
  diagnostics (rolling mean, variance, range), temporal stability (rolling z-scores,
  drift), and environmental features (hour, day, season, temperature).
- **Labels:** None. The model is trained in an unsupervised fashion.

---

## 5. Evaluation Data

- **Validation split:** 2023-07-01 to 2023-12-31.
- **Test split:** 2024-01-01 onwards.
- **Evaluation approach:** Because no verified failure labels exist, evaluation
  uses score distribution analysis, rolling-origin backtest stability, and
  expert review of flagged machines.

---

## 6. Metrics

| Metric | Description | Notes |
|---|---|---|
| Anomaly rate | Fraction of draws flagged (score > 0.5) | Target ≈ contamination parameter (5%) |
| Precision@K | Fraction of true anomalies in top-K scored draws | Requires proxy labels or expert review |
| Recall@K | Fraction of true anomalies captured in top-K | Requires proxy labels or expert review |
| Score stability | Standard deviation of scores across backtest windows | Lower = more stable model |

---

## 7. Quantitative Analysis

| Condition | Anomaly Rate | Notes |
|---|---|---|
| Training set baseline | ~5% | Set by contamination parameter |
| Validation set (expected) | TBD | To be updated after data delivery |
| Test set (expected) | TBD | To be updated after data delivery |

*Quantitative results will be updated in `docs/experiment_log.md` as data becomes available.*

---

## 8. Ethical Considerations

- **No outcome prediction:** This system explicitly excludes draw outcome data.
  It cannot and does not identify patterns in winning numbers.
- **Fairness:** Anomaly flags should be reviewed by qualified engineers.
  Automated flags alone must not be used to make decisions affecting players
  or operators.
- **Transparency:** All feature engineering decisions, leakage controls, and
  modelling choices are documented in this repository.
- **Data minimisation:** Only operationally necessary columns are processed.
  Player data is not ingested.

---

## 9. Caveats and Recommendations

- Anomaly scores are relative to the training period. Significant operational
  changes (e.g. new machine models, venue changes) may cause score drift and
  require model retraining.
- The `contamination` parameter (default 0.05) controls the expected anomaly
  rate and should be tuned based on domain knowledge of typical machine failure
  rates.
- This model should be re-trained periodically (e.g. quarterly) to reflect
  current operating norms.
- A qualified engineer should review all high-risk flags before scheduling
  maintenance.
- **Anomaly scores are not confirmed failure probabilities** without verified
  operational labels linked to actual failure events.
