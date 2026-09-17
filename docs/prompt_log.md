# Prompt Log

**Project:** LD ML 03 — Machine Failure Risk

Record every significant AI-assisted prompt interaction here for transparency
and reproducibility. Include the model used, the purpose, and any modifications
made to the output before use.

---

| Date | Model ID | Prompt Purpose | Useful Output | Rejected Output | Modifications | Verification |
|---|---|---|---|---|---|---|
| 2026-09-14 | Gemini 2.5 Pro | Initial project scaffolding — generate directory structure for LD ML 03 anomaly detection app | Full directory structure with src/, tests/, configs/, docs/, app.py, model_card.md, dataset_card.md, requirements.txt | None rejected | Minor docstring formatting adjustments | Verified all generated files; confirmed leakage safety for rolling windows |
| 2026-09-15 | Claude Sonnet / Qwen3-Coder | Label audit and feature engineering pipeline implementation | Strict shift(1) temporal feature engineering pipeline; chronological splitter (70/15/15) | Supervised failure probability classification proposals | Enforced unsupervised anomaly detection using Isolation Forest & Robust Z-Score due to zero ground-truth failure labels | Executed validation tests in tests/test_features.py; verified no lookahead leakage |
| 2026-09-15 | Claude Sonnet | Model training, evaluation, and baseline comparison | Training script for IsolationForestDetector and ZScoreDetector; benchmark evaluation tables | Complex neural time-series architectures for primary tabular prediction | Configured Isolation Forest as primary model with contamination=0.05 and serialized transformer pipelines | Evaluated holdout performance across chronological train/val/test splits |
| 2026-09-16 | Claude Sonnet 4.6 | Gradio executive UI redesign & multi-tab integration | 6-tab executive Gradio Blocks dashboard with dark styling, telemetry cards, Plotly charts | Static input dropdowns and hardcoded threshold cards | Wired interactive dynamic dropdowns and added high-contrast chart badges | Verified live UI rendering in browser at http://localhost:7860 |
| 2026-09-16 | Gemini Flash | Bug remediation for datetime parsing, reason codes, and recent events | Resilient UTC timestamp parsing, exclusion of lottery numbers from reason codes, recent events table | Overly strict NaT abortion checks in validation step | Replaced strict row rejection with graceful invalid-row omission; excluded ball sets from machine telemetry | Tested CSV inference end-to-end; verified full telemetry card and chart generation |
| 2026-09-17 | Claude Sonnet 4.6 (Thinking) | Hugging Face Space deployment — fix Gradio 5 API schema crash, SSR/CSS errors, file upload styling, 3-step UX guide | Monkeypatch for gradio_client schema bug (additionalProperties:False TypeError); api_name=False on all events; removed ssr from launch; comprehensive CSS dark theme variables; 3-step beginner guide in idle card and left sidebar | ssr=False kwarg (not supported in Gradio 5.16.0); white file-upload box (reverted to dark slate per design decision) | Removed ssr, kept show_api=False; retained dark sidebar; implemented white file-upload box on user request | Tested on live HF Space; verified buttons work, text is readable, charts render, anomaly scoring executes |
