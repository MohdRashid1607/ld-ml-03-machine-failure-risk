# Dataset Card — LD ML 03: Lottery Machine Operational Data

**Dataset ID:** `ld-ml-03-lottery-machine-ops`  
**Version:** 0.1.0  
**Last Updated:** 2026-09-14  
**Task Category:** Anomaly Detection (Unsupervised)  

---

> ⚠️ **DISCLAIMER**  
> This dataset and derived models **do not support prediction of winning
> lottery numbers**. Draw outcome columns (ball numbers, positions) are
> excluded from all feature engineering and model training.

---

## 1. Dataset Summary

This dataset contains historical operational records from lottery draw
machines, capturing mechanical, environmental, and scheduling metadata
for each draw event. It is used exclusively for **unsupervised machine
anomaly detection** — identifying unusual operating patterns that may
warrant maintenance inspection.

No player data, ticket information, or draw outcomes (winning numbers) are
included or processed.

---

## 2. Schema / Data Dictionary

| Column Name | Type | Description |
|---|---|---|
| `game_id` | integer | Unique identifier for the lottery game type |
| `draw_id` | integer | Unique identifier for an individual draw event |
| `machine_id` | string | Identifier for the physical lottery draw machine |
| `ball_set_id` | string | Identifier for the set of balls used in the draw |
| `venue_id` | string | Identifier for the draw venue or studio |
| `draw_local_datetime` | datetime (UTC) | Timestamp of the draw in local time, stored as UTC |
| `temperature_c` | float | Ambient temperature at the venue in degrees Celsius |

*Note: Additional operational columns may be present in source data but
are not guaranteed to be populated across all records.*

---

## 3. Source and Provenance

- **Data owner:** Lottery operations authority (internal).
- **Collection method:** Automated draw management system logs, exported
  to CSV format.
- **Geographic scope:** Single jurisdiction (details withheld for data
  governance reasons).
- **Time range:** Historical records spanning multiple years; exact range
  TBD pending data delivery.

---

## 4. Processing Steps

1. **Loading:** CSV ingestion via `src/data/loader.py` with schema validation.
2. **Datetime parsing:** `draw_local_datetime` parsed to UTC-aware Timestamp.
3. **Chronological sorting:** Records sorted by `draw_local_datetime` before
   any processing.
4. **Missing value imputation:** Numeric columns imputed with column median;
   categorical columns filled with `"__missing__"` placeholder.
5. **Encoding:** Categorical features label-encoded; numeric features
   standard-scaled (fit on training split only).
6. **Feature engineering:** Machine history, rolling distribution diagnostics,
   temporal stability, and environmental features added (all leakage-safe).

---

## 5. Splits

| Split | Date Boundary | Purpose |
|---|---|---|
| Training | ≤ 2023-06-30 | Model fitting |
| Validation | 2023-07-01 – 2023-12-31 | Hyperparameter selection, threshold tuning |
| Test | ≥ 2024-01-01 | Final evaluation |

Splits are strictly chronological (no shuffling) to prevent temporal leakage.

---

## 6. License Notes

- The dataset is proprietary to the lottery operations authority.
- It may not be redistributed, published, or used for purposes other than
  internal operational analytics without written approval.
- This repository does not contain or commit the raw data; only the processing
  and modelling code is version-controlled.

---

## 7. Known Gaps

| Gap | Impact | Mitigation |
|---|---|---|
| No verified machine failure / maintenance event labels | Precludes supervised classification | Unsupervised anomaly detection used instead |
| Incomplete temperature records in some periods | Rolling temperature features may be NaN | Median imputation applied; missingness flag added |
| Unknown ball set retirement / replacement history | Service interval features may be inaccurate | `time_since_last_service` flagged with NaN for first draw per machine |
| Single jurisdiction data | Model may not generalise to other lotteries | Document scope limitation; re-train for new jurisdictions |
