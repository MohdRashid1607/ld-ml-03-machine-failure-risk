# Experiment Log

**Project:** LD ML 03 — Machine Failure Risk
**Last Updated:** 2026-09-17

Record every significant experiment here. Include enough detail to reproduce results.

---

| Date | Model | Config | Train Anomaly Rate | Val Anomaly Rate | Val Score Mean | Val Score Std | Notes |
|---|---|---|---|---|---|---|---|
| 2026-09-15 | Robust Z-Score | threshold=3σ, median+MAD | — | **100%** (all flagged) | — | — | Rejected as primary: unacceptable false-alert burden. Retained as mandatory baseline comparison. |
| 2026-09-15 | Isolation Forest | contamination=0.05, n_estimators=100, random_state=42 | **4.84%** | **0.00%** | 0.0175 | ±0.0218 | **Selected primary model.** 0% false-alert rate on val and test holdouts. Max val score: 0.0635. Test score mean: 0.0214, std: ±0.0154, max: 0.0580. 212 features. 124 train rows. |
