# CHANGELOG

All notable changes to this project are documented here.

---

## [0.1.0] — 2026-09-17

### Added
- **Gradio Space deployed** to `khalidml65/lottery-machine-failure-risk` on Hugging Face
- 6-tab executive dashboard: Anomaly Detection, Data Quality, Model Performance, Model Card, Dataset Card, Responsible Use
- 3-step Quick Start guide in the idle welcome card and left sidebar
- Comprehensive dark-theme CSS with CSS variables for Gradio 5 (`--tw-prose-*`)
- `data/sample_draws.csv` — clean sample file without spaces in filename (HF-compatible)
- `deploy_space.py` — single-command deployment script for HF Space updates
- `docs/research_memo.md` — full ML research documentation
- `docs/evaluation_report.md` — chronological evaluation and error analysis report
- `LICENSE` — MIT License file
- `models/README.md` — expanded inference guide with example code

### Fixed
- **Gradio 5 schema crash:** Monkeypatched `gradio_client.utils._json_schema_to_python_type` to handle `additionalProperties: False` (boolean) without TypeError
- **`ssr=False` crash:** Removed unsupported kwarg from `app.launch()` for Gradio 5.16.0
- **`css=CSS` crash:** Moved `css` argument from `app.launch()` to `gr.Blocks()` (Gradio 5 requirement)
- **Buttons not responding:** Added `api_name=False` to all `.click()` and `.change()` event listeners; added `show_api=False` to launch
- **Black text on dark background:** Added comprehensive `--tw-prose-*` CSS variables and `[data-testid="markdown"] *` selectors
- **Future timestamp rejection:** Removed overly strict datetime validation that blocked valid test datasets
- File upload box made solid white with dark-grey visible text

### Changed
- `requirements.txt` — pinned `gradio==5.16.0`, added `huggingface_hub>=0.28.0`, `httpx`, `requests`
- `models/training_manifest.json` — changed absolute Windows paths to relative paths for cross-platform reproducibility
- `docs/experiment_log.md` — filled with real benchmark results (Isolation Forest vs Z-Score)
- `docs/prompt_log.md` — added 6th session entry for 2026-09-17 deployment session

---

## [0.0.2] — 2026-09-16

### Added
- Gradio app initial construction (`app.py`) with 6 tabs
- Dark executive CSS with Inter font and Plotly chart styling
- Dynamic dropdowns for game and machine filter
- Anomaly trend chart and score distribution chart (Plotly)
- Top 5 reason codes card with feature attribution
- Recent operational events table

### Fixed
- Datetime UTC parsing with `errors='coerce'` for mixed-timezone inputs
- Lottery ball numbers excluded from reason code feature attribution

---

## [0.0.1] — 2026-09-15

### Added
- `src/data/` — loader, validator, splitter, transformer
- `src/features/` — machine_history, distribution_diagnostics, temporal_stability, environment, data_quality, pipeline
- `src/models/` — isolation_forest, baseline_zscore, trainer, scoring
- `src/evaluation/` — backtest, metrics, plots
- `tests/` — test_data, test_features, test_models, test_ui
- `configs/` — data_config.yaml, model_config.yaml, thresholds.yaml
- `docs/` — label_audit.md, model_selection_memo.md, experiment_log.md, prompt_log.md
- Trained Isolation Forest (212 features, 124 training draws, contamination=0.05)
- Hugging Face Dataset and Model repositories published

---

## [0.0.0] — 2026-09-14

### Added
- Initial project scaffolding
- Label audit: confirmed no verified failure labels exist → unsupervised anomaly detection selected
- `model_card.md`, `dataset_card.md`, `README.md`
- `.gitignore` with credential and model artefact exclusions
