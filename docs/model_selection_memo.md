# Model Selection Memo

**Project:** LD ML 03 — Machine Failure Risk  
**Date:** 2026-09-14  
**Status:** Approved  

---

## 1. Problem Statement

Lottery draw machines exhibit complex mechanical and environmental interactions
that can lead to irregular operational patterns before or after a failure event.
The objective of this project is to detect *statistically abnormal* machine
operating patterns from historical draw and environmental data, enabling
operators to prioritise inspection of flagged machines.

Because no verified failure or maintenance event labels exist in the current
dataset (see `docs/label_audit.md`), the problem is framed as **unsupervised
anomaly detection** rather than supervised failure prediction.

---

## 2. Target Definition

| Attribute | Description |
|---|---|
| **Task type** | Unsupervised anomaly detection |
| **Output** | Anomaly score ∈ [0, 1] and binary flag |
| **NOT the output** | Failure probability, draw outcome, or winning number prediction |
| **Positive class** | High-anomaly-score machine draw (score > threshold) |

---

## 3. Sample Size Estimates

| Split | Date Range | Approximate Row Count |
|---|---|---|
| Training | Up to 2023-06-30 | *TBD — depends on data delivery* |
| Validation | 2023-07-01 to 2023-12-31 | *TBD* |
| Test | 2024-01-01 onwards | *TBD* |

*Note: Row counts will be updated once the full dataset is received.*

---

## 4. Shortlisted Approaches

| Approach | Rationale | Limitations |
|---|---|---|
| **Robust Z-Score Baseline** | Simple, interpretable, parameter-free. Uses median/MAD to resist outlier distortion. Serves as a sanity-check baseline. | Assumes univariate independence; misses cross-feature correlations. |
| **Isolation Forest** | Tree-based ensemble; does not assume Gaussian distributions; scales well; natively handles multivariate data; well-suited to tabular anomaly detection. | Sensitive to `contamination` parameter; interpretation less direct than z-scores. |
| **Granite TTM R2.1 (Time-Series Foundation Model)** | Pre-trained transformer for time-series anomaly; zero-shot or few-shot capability; may generalise well without labelled data. | Large computational footprint; requires careful prompting; output interpretability limited; currently designated as *future exploration*. |

---

## 5. Final Model Selection

**Primary model: Isolation Forest**

Rationale:
- Best balance of detection power and computational cost for tabular data.
- Does not require labelled anomalies for training.
- Contamination parameter (`0.05`) is tunable and interpretable.
- Well-supported by scikit-learn with deterministic reproducibility via `random_state`.

**Baseline: Robust Z-Score Detector**

Retained as a mandatory comparison benchmark. Any release candidate must
outperform the z-score baseline on the validation set.

---

## 6. Limitations

- Without verified labels, precision and recall cannot be computed against
  true failures; evaluation relies on proxy metrics (Precision@K, score distributions).
- Anomaly scores are relative — they indicate deviation from training-period norms,
  not absolute failure risk.
- Seasonal and environmental shifts may inflate false-positive rates;
  regular model re-training is recommended.
- **This system does not predict lottery draw outcomes or winning numbers.**
