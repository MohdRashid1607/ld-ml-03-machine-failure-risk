# Label Audit Report

**Project:** LD ML 03 — Machine Failure Risk  
**Date:** 2026-09-14  
**Auditor:** Project Team  

---

## 1. Purpose

This document records the outcome of a systematic audit to determine whether
verified, trustworthy machine failure or maintenance event labels exist in the
current dataset. The presence of such labels would enable supervised
classification or regression approaches.

---

## 2. Audit Findings

After reviewing all available data sources, **no verified machine
failure or maintenance event labels were found** in the current dataset.

Specifically:

- No column records confirmed machine faults, stoppages, or unplanned downtime.
- No external maintenance logs were available for join.
- No structured operational event stream (e.g. SCADA alerts) was provided.
- Draw-level metadata does not include technician sign-off or fault codes.

---

## 3. Consequence for Modelling Approach

Because no trustworthy event labels exist, **supervised failure-prediction
tasks are not permissible** with this dataset.  Using heuristic proxies
(e.g. flagging draws where a machine was serviced shortly afterwards) as
pseudo-labels would introduce severe look-ahead bias and produce unreliable
probability estimates.

The project therefore uses **unsupervised anomaly detection**:

- **Primary model:** Isolation Forest
- **Baseline model:** Robust Z-Score Detector

Anomaly scores reflect statistical deviations from normal operating patterns,
not confirmed failure probabilities.

---

## 4. Permitted Tasks Summary

| Data Condition | Permitted Task | Target |
|---|---|---|
| No trustworthy event labels | Unsupervised anomaly detection | Anomaly score and review flag; no failure probability |
| Verified maintenance logs available (future) | Supervised binary classification | P(failure within N draws) |
| Time-to-event labels available (future) | Survival analysis / time-to-failure | Expected draws until next failure |

---

## 5. Recommendations

1. **Operational labelling:** Engage with lottery machine operators to obtain
   structured maintenance logs with timestamps, fault codes, and resolution
   records.
2. **Proxy label review:** If proxy labels (e.g. "machine replaced within
   30 draws") are later considered, they must undergo a separate bias-risk
   assessment before use in model training.
3. **Ongoing monitoring:** Re-run this audit whenever a new data source is
   onboarded.

---

> ⚠️ **This system does not predict winning numbers. Anomaly scores are
> not confirmed failure probabilities without verified operational labels.**
