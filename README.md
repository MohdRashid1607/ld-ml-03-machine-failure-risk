---
title: LD ML 03 Lottery Machine Failure Risk Application
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.16.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# LD ML 03 — Lottery Draw Machine Failure Risk & Telemetry Monitor

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Space-khalidml65/lottery--machine--failure--risk-yellow)](https://huggingface.co/spaces/khalidml65/lottery-machine-failure-risk)
[![HF Dataset](https://img.shields.io/badge/🤗%20Dataset-khalidml65/lottery--draw--machine--telemetry-blue)](https://huggingface.co/datasets/khalidml65/lottery-draw-machine-telemetry)
[![HF Model](https://img.shields.io/badge/🤗%20Model-khalidml65/lottery--isolation--forest--detector-green)](https://huggingface.co/khalidml65/lottery-isolation-forest-detector)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **⚠️ Responsible Use Notice:** This application assesses machine anomaly risk from operational patterns. It does **not** predict winning lottery numbers. Anomaly scores are not confirmed failure probabilities without verified operational labels.

---

## Overview

This project is **Trainee Homework Assignment LD ML 03**: a reproducible machine-learning application deployed on Hugging Face that detects abnormal operating patterns in lottery draw machines using **unsupervised anomaly detection**.

### Label Gate Decision
A formal label audit (see `docs/label_audit.md`) confirmed that **no verified machine failure or maintenance labels exist** in the dataset. The project therefore uses:

- **Primary Model:** Isolation Forest (multivariate unsupervised anomaly detection)
- **Baseline Model:** Robust Z-Score Detector (median + MAD) — mandatory comparison
- **Output:** Anomaly score ∈ [0,1] and risk flag — NOT a failure probability

---

## Hugging Face Repositories

| Repository | Link | Purpose |
|---|---|---|
| 🎯 Gradio Space | [khalidml65/lottery-machine-failure-risk](https://huggingface.co/spaces/khalidml65/lottery-machine-failure-risk) | Live interactive dashboard |
| 📦 Dataset | [khalidml65/lottery-draw-machine-telemetry](https://huggingface.co/datasets/khalidml65/lottery-draw-machine-telemetry) | Versioned processed telemetry data |
| 🤖 Model | [khalidml65/lottery-isolation-forest-detector](https://huggingface.co/khalidml65/lottery-isolation-forest-detector) | Trained Isolation Forest model + transforms |

---

## Project Structure

```
ld-ml-03-machine-failure-risk/
│
├── app.py                    # Gradio Space entry point (6-tab dashboard)
├── requirements.txt          # Pinned dependencies
├── model_card.md             # Model documentation (intended use, evaluation, ethics)
├── dataset_card.md           # Dataset documentation (schema, provenance, splits)
├── CHANGELOG.md              # Version history
├── LICENSE                   # MIT License
├── README.md                 # This file
│
├── src/
│   ├── data/                 # loader.py, validator.py, splitter.py, transformer.py
│   ├── features/             # machine_history, distribution_diagnostics, temporal_stability,
│   │                         # environment, data_quality, pipeline
│   ├── models/               # isolation_forest, baseline_zscore, trainer, scoring
│   └── evaluation/           # backtest, metrics, plots
│
├── models/
│   ├── iso_forest_detector.joblib    # Trained Isolation Forest (212 features)
│   ├── zscore_detector.joblib        # Trained Z-Score baseline
│   ├── transformers/                 # Fitted preprocessing pipeline
│   ├── training_manifest.json        # Config, splits, feature count
│   └── README.md                     # Inference guide with code example
│
├── data/
│   ├── raw/                  # Source CSV files (excluded from Git via .gitignore)
│   ├── processed/            # train_processed.csv, val_processed.csv, test_processed.csv
│   └── sample_draws.csv      # Clean sample for Space testing
│
├── tests/
│   ├── test_data.py          # Data loading and validation tests
│   ├── test_features.py      # Feature engineering and leakage tests
│   ├── test_models.py        # Model training and scoring tests
│   └── test_ui.py            # Gradio UI function tests
│
├── configs/
│   ├── data_config.yaml      # Data paths and schema settings
│   ├── model_config.yaml     # Model hyperparameters
│   └── thresholds.yaml       # Anomaly score thresholds
│
├── docs/
│   ├── label_audit.md        # Label gate decision (no failure labels found)
│   ├── model_selection_memo.md # 3-approach comparison; Isolation Forest selected
│   ├── research_memo.md      # Full ML research documentation
│   ├── evaluation_report.md  # Chronological evaluation & error analysis
│   ├── experiment_log.md     # Per-experiment benchmark results
│   └── prompt_log.md         # LLM-assisted development log (6 sessions)
│
├── deploy_space.py           # One-command HF Space deployment script
├── run_pipeline.py           # Re-run full training pipeline from scratch
└── run_evaluation.py         # Re-run holdout evaluation
```

---

## Quick Start — Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Gradio app
python app.py
# → Open http://localhost:7860
```

## Running Tests

```bash
pytest tests/ -v
```

## Re-Training the Model

```bash
# Place your raw CSV in data/raw/ then:
python run_pipeline.py
```

---

## Uploading Your Own CSV

The Space accepts any CSV with these mandatory columns:

| Column | Type | Description |
|---|---|---|
| `game_id` | string | Lottery game identifier |
| `draw_id` | string | Unique draw identifier |
| `machine_id` | string | Machine identifier |
| `ball_set_id` | string | Ball set identifier |
| `venue_id` | string | Venue identifier |
| `draw_local_datetime` | datetime | Local draw date and time |
| `temperature_c` | float | Ambient temperature at draw time (°C) |

Additional columns (humidity, latitude, longitude, etc.) will be used automatically if present.

---

## Benchmark Results

| Model | Holdout Anomaly Rate | Val Score Mean | Val Score Std | Decision |
|---|---|---|---|---|
| Robust Z-Score (baseline) | **100%** (all flagged) | — | — | ❌ Rejected — unacceptable false-alert burden |
| **Isolation Forest (primary)** | **0.00%** | 0.0175 | ±0.0218 | ✅ Selected — 0% false alerts on holdouts |

Isolation Forest training anomaly rate: **4.84%** (target contamination: 5.0%)  
Test holdout: mean score 0.0214, std ±0.0154, max 0.0580 — 0.00% flagged.

---

## Label Audit Summary

No verified machine failure or maintenance labels were found. See [`docs/label_audit.md`](docs/label_audit.md) for the full audit. The project uses **unsupervised anomaly detection** accordingly.

---

## Documentation Index

| Document | Purpose |
|---|---|
| [`docs/label_audit.md`](docs/label_audit.md) | Label gate decision — why unsupervised approach was chosen |
| [`docs/model_selection_memo.md`](docs/model_selection_memo.md) | 3-approach comparison and final model selection justification |
| [`docs/research_memo.md`](docs/research_memo.md) | Full ML research process, data overview, feature engineering |
| [`docs/evaluation_report.md`](docs/evaluation_report.md) | Chronological evaluation, error analysis, threshold sensitivity |
| [`docs/experiment_log.md`](docs/experiment_log.md) | Per-experiment benchmark results table |
| [`docs/prompt_log.md`](docs/prompt_log.md) | LLM-assisted development log (6 sessions across 4 days) |
| [`model_card.md`](model_card.md) | Model intended use, ethics, limitations, evaluation |
| [`dataset_card.md`](dataset_card.md) | Dataset schema, provenance, processing, splits, known gaps |
| [`CHANGELOG.md`](CHANGELOG.md) | Version history and change log |
| [`models/README.md`](models/README.md) | Inference guide with Python code example |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

> **⚠️ This system does not predict lottery draw outcomes or winning numbers. All anomaly scores are statistical deviations from historical operating norms and require qualified engineering review before operational decisions are made.**