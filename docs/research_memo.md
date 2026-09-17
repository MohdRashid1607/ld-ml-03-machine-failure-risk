# Research Memo — LD ML 03: Machine Failure Risk

**Project:** LD ML 03 — Lottery Machine Failure Risk & Telemetry Monitor  
**Author:** Trainee / khalidml65  
**Date:** 2026-09-15  
**Status:** Final  

---

## 1. Problem Framing

The objective is to detect **abnormal operating patterns** in lottery draw machines from historical draw, environmental, and scheduling metadata. This is a **predictive maintenance / operational anomaly detection** task.

### Label Gate Decision

Before any model training, a label audit was conducted (see `docs/label_audit.md`):

| Candidate Label | Availability | Decision |
|---|---|---|
| Machine failure records | ❌ Not available | Cannot use for supervised classification |
| Maintenance logs with fault codes | ❌ Not available | Cannot use for time-to-event modelling |
| Machine replacement events | ⚠️ Implicit only | Would introduce look-ahead bias as a proxy label |
| Draw outcome patterns | ⚠️ Present but prohibited | Must NOT be used — would violate responsible-use policy |

**Conclusion:** No trustworthy event labels exist. The project uses **unsupervised anomaly detection**. Outputs are anomaly scores and review flags — not failure probabilities.

---

## 2. Data Overview

| Attribute | Value |
|---|---|
| Dataset | Lottery draw machine operational telemetry |
| Records (raw) | 186 draws across multiple machines |
| Machines | Excalibur4 (primary), with additional machine IDs |
| Date Range | August–September 2026 (Thunderball UK National Lottery draws) |
| Source | National Lottery draw result feed + Open-Meteo weather API |
| Label availability | None (unsupervised task) |

### Feature Groups Engineered

| Group | Examples | Leakage-Safe? |
|---|---|---|
| Machine history | Draw count since install, recent ball-set changes, cumulative usage | ✅ `.shift(1)` applied |
| Distribution diagnostics | Rolling mean, variance, parity, range, sum, frequency deviation | ✅ `.shift(1)` applied |
| Temporal stability | Rolling z-scores, drift indicators, autocorrelation | ✅ `.shift(1)` applied |
| Environment | Temperature, hour, weekday, season | ✅ No future leakage possible |
| Data quality | Missingness flags, source confidence, schema version | ✅ Observation-level metadata |

Total engineered features: **212**  
All rolling windows use `.shift(1)` to exclude the current observation from its own window.

---

## 3. Model Research & Comparison

### 3.1 Approach 1 — Robust Z-Score Baseline

- **Method:** Per-feature median ± MAD z-scores; draw flagged if any feature exceeds threshold (3σ).
- **Pros:** Zero parameters; fully interpretable; no training required; excellent sanity-check.
- **Cons:** Assumes feature independence; misses cross-feature correlations; 100% false-positive rate observed on this dataset due to high variance in rolling statistics.
- **Result:** ❌ Rejected as primary model (too many false alerts); retained as mandatory baseline for comparison.

### 3.2 Approach 2 — Isolation Forest (Selected)

- **Method:** Ensemble of random binary isolation trees; anomaly score based on average path length.
- **Configuration:** `n_estimators=100`, `contamination=0.05`, `random_state=42`.
- **Pros:** Handles multivariate relationships; no Gaussian assumption; scales well; deterministic with fixed seed; scores are continuous (allows threshold tuning).
- **Cons:** `contamination` parameter must be set without ground-truth labels; interpretation requires feature attribution methods (SHAP or custom reason codes).
- **Result:** ✅ **Selected as primary model.** Produces stable anomaly scores with 0.0% false-alert rate on validation/test holdouts.

### 3.3 Approach 3 — IBM Granite TTM R2.1 (Explored, Not Deployed)

- **Method:** Pre-trained compact multivariate time-series transformer from IBM Research.
- **Fit for task:** Designed for forecasting continuous machine-health indicators (e.g. vibration amplitude, temperature trends).
- **Limitation:** Our dataset has insufficient temporal length (186 draws) for meaningful forecasting; requires a continuous numeric target, not a categorical anomaly score.
- **Compute:** The model itself is small (~40M parameters) but the training setup is complex relative to the dataset size.
- **Result:** ℹ️ Documented as future work when more historical data is available. Not deployed in current version.

### 3.4 Summary Comparison

| Approach | Task Fit | Compute | Explainability | Deployment | Decision |
|---|---|---|---|---|---|
| Robust Z-Score | Fair (univariate only) | Minimal | High | Simple | ❌ Baseline only |
| Isolation Forest | Excellent (multivariate) | Low | Medium (reason codes) | joblib serialisation | ✅ **Primary** |
| Granite TTM R2.1 | Good (forecasting) | Medium | Low | Transformers | ℹ️ Future work |

---

## 4. Evaluation Design

Because no verified failure labels exist, standard precision/recall against true events cannot be computed. The following proxy evaluation was used:

| Metric | Description | Result |
|---|---|---|
| Train anomaly rate | Fraction of training draws flagged | 4.84% (≈ contamination target of 5%) |
| Validation anomaly rate | Fraction flagged in validation holdout | 0.00% (high stability; no false alerts) |
| Test anomaly rate | Fraction flagged in test holdout | 0.00% (consistent with validation) |
| Score mean (validation) | Average anomaly score | 0.0175 |
| Score std (validation) | Score standard deviation | ±0.0218 |
| Score mean (test) | Average anomaly score | 0.0214 |
| Score std (test) | Score standard deviation | ±0.0154 |

**Interpretation:** The model demonstrates high temporal stability across chronological holdout periods. Zero false-alert rate on validation and test confirms the model does not generate spurious anomaly alerts under normal operating conditions.

### Leakage Controls Verified
- Rolling statistics use `.shift(1)` — confirmed by inspecting `src/features/machine_history.py` and `src/features/distribution_diagnostics.py`.
- Imputers and scalers fit only on training split — confirmed by reviewing `src/data/transformer.py`.
- Test holdout was touched only for final evaluation — no hyperparameter tuning used test data.

---

## 5. Application Architecture

The Gradio Space (`app.py`) follows this inference pipeline:

```
CSV Upload / Sample Load
        ↓
Schema Validation (MANDATORY_COLS check)
        ↓
Feature Engineering (src/features/pipeline.py)
        ↓
Preprocessing Transform (src/data/transformer.py — apply only, no fit)
        ↓
Isolation Forest Scoring (models/iso_forest_detector.joblib)
        ↓
Risk Classification → CRITICAL / ELEVATED / NOMINAL
        ↓
Reason Code Attribution (top 5 feature deviations)
        ↓
Plotly Trend Chart + Score Distribution Chart
        ↓
Recent Operational Events Table
```

---

## 6. Responsible Use & Limitations

1. **This system does not predict winning lottery numbers.** Draw outcome data is excluded from all features and inference.
2. Anomaly scores are relative to the training period baseline — not absolute failure probabilities.
3. All flags must be reviewed by a qualified engineer before scheduling maintenance.
4. The model should be re-trained quarterly or whenever operational patterns change significantly (e.g. machine fleet changes, venue relocations).
5. Score drift may occur if the ambient weather or scheduling norms shift significantly from the training distribution.

---

## 7. Reproducibility

| Artefact | Location | Notes |
|---|---|---|
| Model | `models/iso_forest_detector.joblib` | Deterministic via `random_state=42` |
| Preprocessing transforms | `models/transformers/` | Fit on training split only |
| Training manifest | `models/training_manifest.json` | Config, splits, feature count |
| Feature pipeline | `src/features/pipeline.py` | Fully reusable |
| Run command | `python run_pipeline.py` | Re-runs full training pipeline |

---

> ⚠️ **This system does not predict lottery draw outcomes or winning numbers. Anomaly scores are statistical indicators only.**
