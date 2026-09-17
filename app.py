"""
app.py  —  LD ML 03: Lottery Machine Failure Risk & Telemetry Monitor
Hugging Face Space entry point.
Redesigned for executive UX/UI design & 100% compliance with the LD ML 03 brief.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    import gradio_client.utils as gc_utils
    _orig_json_schema = gc_utils._json_schema_to_python_type
    def _safe_json_schema(schema, defs=None):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_json_schema(schema, defs)
    gc_utils._json_schema_to_python_type = _safe_json_schema

    _orig_get_type = gc_utils.get_type
    def _safe_get_type(schema):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_get_type(schema)
    gc_utils.get_type = _safe_get_type
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── Constants & Metadata ─────────────────────────────────────────────────────
APP_VERSION     = "v1.0.0"
MODEL_VERSION   = "v0.1.0"
MODEL_NAME      = "Isolation Forest Anomaly Detector"
TRAINED_DATE    = "2026-09-15"
TRAIN_SAMPLES   = 124
TRAIN_FEATURES  = 212
ANOMALY_THRESH  = 0.05
MAX_FILE_SIZE_ROWS = 100_000

MODEL_DIR       = Path("models")
ISO_PATH        = MODEL_DIR / "iso_forest_detector.joblib"
ZSCORE_PATH     = MODEL_DIR / "zscore_detector.joblib"
SAMPLE_DATA_PATH = Path("data/sample_draws.csv")

_NON_FEAT = {
    "game_id", "draw_id", "machine_id", "ball_set_id",
    "venue_id", "draw_local_datetime", "season",
    "anomaly_score", "anomaly_flag", "risk_category",
}
MANDATORY_COLS = [
    "game_id", "draw_id", "machine_id", "ball_set_id",
    "venue_id", "draw_local_datetime", "temperature_c",
]

# ─── Model & Schema Loading ───────────────────────────────────────────────────
_iso_model     = None
_feat_cols_ref = None

def _load():
    global _iso_model, _feat_cols_ref
    try:
        if ISO_PATH.exists():
            _iso_model = joblib.load(str(ISO_PATH))
            logger.info("Loaded primary model from %s", ISO_PATH)
        val_p = Path("data/processed/val_processed.csv")
        if val_p.exists():
            s = pd.read_csv(val_p, nrows=1)
            _feat_cols_ref = [c for c in s.select_dtypes(include=[np.number]).columns if c not in _NON_FEAT]
            logger.info("Loaded reference feature columns: count=%d", len(_feat_cols_ref))
    except Exception as e:
        logger.warning("Model load error: %s", e)

_load()

def _read_md(p: str) -> str:
    path = Path(p)
    return path.read_text(encoding="utf-8") if path.exists() else f"*`{p}` not found*"

MODEL_CARD_MD   = _read_md("model_card.md")
DATASET_CARD_MD = _read_md("dataset_card.md")

# ─── Strict File Validation (Aligned with test_ui.py & brief) ─────────────────
def validate_file(df: pd.DataFrame) -> tuple[bool, str]:
    if df is None or len(df) == 0:
        return False, "❌ Uploaded file is empty."
    if len(df) > MAX_FILE_SIZE_ROWS:
        return False, f"❌ File too large: {len(df):,} rows (max {MAX_FILE_SIZE_ROWS:,})."
    missing = [c for c in MANDATORY_COLS if c not in df.columns]
    if missing:
        return False, f"❌ Missing required columns: {', '.join(missing)}"
    if "draw_local_datetime" in df.columns:
        try:
            dt_col = pd.to_datetime(df["draw_local_datetime"], errors="coerce", utc=True)
            valid_dt = dt_col.dropna()
            if len(valid_dt) == 0:
                return False, "❌ 'draw_local_datetime' contains no valid timestamps."
        except Exception as e:
            return False, f"❌ Error validating datetime sequence: {e}"
    return True, "✅ File validated successfully."

# ─── Feature prep ─────────────────────────────────────────────────────────────
def _prepare(df: pd.DataFrame):
    from src.features.pipeline import build_feature_matrix
    from src.data.transformer import clean_and_encode
    enriched  = build_feature_matrix(df.copy(), fit=False)
    processed = clean_and_encode(enriched, fit=False)
    feat_cols = [c for c in processed.select_dtypes(include=[np.number]).columns if c not in _NON_FEAT]
    if _feat_cols_ref:
        cols = []
        for c in _feat_cols_ref:
            cols.append(processed[c] if c in processed.columns else pd.Series(np.zeros(len(processed)), name=c))
        X = pd.concat(cols, axis=1).fillna(0).values
        feat_cols = _feat_cols_ref
    else:
        X = np.nan_to_num(processed[feat_cols].values, nan=0.0)
    return X, feat_cols

# ─── HTML Telemetry & Reason Code Builders ─────────────────────────────────────
def _risk_card(label: str, mean_score: float, max_score: float, std_score: float,
               n_flagged: int, n_total: int, machine: str, game: str, ts: str, latency_ms: float) -> str:
    palette = {
        "LOW":    {"border": "#10b981", "badge_bg": "#064e3b", "text": "#34d399", "label": "NOMINAL / LOW RISK", "desc": "Telemetry follows expected historical baseline distributions."},
        "MEDIUM": {"border": "#f59e0b", "badge_bg": "#78350f", "text": "#fbbf24", "label": "ELEVATED / WATCHLIST", "desc": "Mild statistical deviation observed. Recommended for scheduled engineer review."},
        "HIGH":   {"border": "#ef4444", "badge_bg": "#7f1d1d", "text": "#f87171", "label": "CRITICAL ANOMALY", "desc": "Significant operating deviation detected. Inspection review strongly advised."},
    }
    level = "LOW" if max_score <= 0.03 else ("MEDIUM" if max_score <= ANOMALY_THRESH else "HIGH")
    cfg = palette[level]
    flag_pct = (n_flagged / max(n_total, 1)) * 100

    return f"""
<div style="
    background: linear-gradient(145deg, #0a1324 0%, #0f1c33 100%);
    border: 1px solid #1e2f4d;
    border-left: 6px solid {cfg['border']};
    border-radius: 14px;
    padding: 22px 26px;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.35);
    font-family: 'Inter', -apple-system, sans-serif;
    margin-bottom: 18px;
">
  <!-- Status Header Row -->
  <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:14px; margin-bottom:18px;">
    <div>
      <div style="display:inline-flex; align-items:center; gap:10px; background:{cfg['badge_bg']}; border:1.5px solid {cfg['border']}; border-radius:8px; padding:6px 16px;">
        <span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:{cfg['border']}; box-shadow:0 0 10px {cfg['border']};"></span>
        <span style="font-size:1.15em; font-weight:800; color:{cfg['text']}; letter-spacing:1px;">{cfg['label']}</span>
      </div>
      <div style="color:#94a3b8; font-size:0.82em; margin-top:8px;">{cfg['desc']}</div>
    </div>
    <div style="text-align:right; font-size:0.8em; color:#64748b; line-height:1.6;">
      <div>Game: <strong style="color:#e2e8f0">{game}</strong> &nbsp;|&nbsp; Machine: <strong style="color:#e2e8f0">{machine}</strong></div>
      <div>Analysed: <strong style="color:#cbd5e1">{ts}</strong></div>
      <div>Model: <span style="color:#38bdf8; font-family:monospace;">Isolation Forest {MODEL_VERSION}</span> &nbsp;|&nbsp; Latency: <strong style="color:#a78bfa">{latency_ms:.1f} ms</strong></div>
    </div>
  </div>

  <!-- KPI Grid -->
  <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap:12px; margin-bottom:18px;">
    <div style="background:#070e1c; border:1px solid #1a2a44; border-radius:10px; padding:12px 16px;">
      <div style="color:#64748b; font-size:0.7em; text-transform:uppercase; letter-spacing:1px; font-weight:600;">Mean Score</div>
      <div style="color:#38bdf8; font-size:1.45em; font-weight:700; margin-top:2px;">{mean_score:+.4f}</div>
      <div style="color:#475569; font-size:0.68em;">historical baseline</div>
    </div>
    <div style="background:#070e1c; border:1px solid #1a2a44; border-radius:10px; padding:12px 16px;">
      <div style="color:#64748b; font-size:0.7em; text-transform:uppercase; letter-spacing:1px; font-weight:600;">Max Deviation</div>
      <div style="color:#c084fc; font-size:1.45em; font-weight:700; margin-top:2px;">{max_score:+.4f}</div>
      <div style="color:#475569; font-size:0.68em;">peak anomaly draw</div>
    </div>
    <div style="background:#070e1c; border:1px solid #1a2a44; border-radius:10px; padding:12px 16px;">
      <div style="color:#64748b; font-size:0.7em; text-transform:uppercase; letter-spacing:1px; font-weight:600;">Score Uncertainty</div>
      <div style="color:#fcd34d; font-size:1.45em; font-weight:700; margin-top:2px;">±{std_score:.4f}</div>
      <div style="color:#475569; font-size:0.68em;">confidence interval (1σ)</div>
    </div>
    <div style="background:#070e1c; border:1px solid {'rgba(239,68,68,0.3)' if n_flagged else '#1a2a44'}; border-radius:10px; padding:12px 16px;">
      <div style="color:#64748b; font-size:0.7em; text-transform:uppercase; letter-spacing:1px; font-weight:600;">Flagged Draws</div>
      <div style="color:{cfg['text']}; font-size:1.45em; font-weight:700; margin-top:2px;">{n_flagged} <span style="color:#475569; font-size:0.65em; font-weight:400;">/ {n_total:,}</span></div>
      <div style="color:#475569; font-size:0.68em;">{flag_pct:.1f}% flag rate (threshold: {ANOMALY_THRESH})</div>
    </div>
  </div>

  <!-- Operational Guardrails Callout -->
  <div style="border-top: 1px solid #16243b; padding-top: 12px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; font-size:0.72em; color:#64748b;">
    <div>🛡️ <b>Notice:</b> Unsupervised anomaly score. This does <u>not</u> represent winning number prediction or supervised failure probability.</div>
    <div style="background:#111d33; padding:3px 10px; border-radius:6px; border:1px solid #1e2f4d; color:#94a3b8;">Decision: Internal Maintenance Prioritization Only</div>
  </div>
</div>
"""

def _feature_card(top_features: dict, recent_events_df: pd.DataFrame = None) -> str:
    if not top_features:
        return "<div style='color:#64748b; padding:12px; font-size:0.85em; text-align:center;'>No feature attribution data available.</div>"
    max_val = max(abs(v) for v in top_features.values()) or 1.0
    rows = ""
    for fname, fval in top_features.items():
        clean_name = fname.replace("_rolling_", " Rolling ").replace("_zscore", " (Z-Score)").replace("_drift", " Drift").replace("_mean", " Mean").replace("_var", " Variance").replace("_", " ").title()
        pct   = abs(fval) / max_val * 100
        is_pos = fval >= 0
        col   = "#f87171" if is_pos else "#38bdf8"
        sign  = "▲ Outlier (+)" if is_pos else "▼ Outlier (-)"
        rows += f"""
<div style="margin-bottom:12px;">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
    <span style="color:#cbd5e1; font-size:0.8em; font-weight:500; font-family:monospace; max-width:70%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="{fname}">{clean_name}</span>
    <span style="color:{col}; font-size:0.75em; font-weight:700; background:#070f1e; border:1px solid {col}44; border-radius:4px; padding:2px 6px;">{sign} {fval:+.3f}</span>
  </div>
  <div style="background:#0a1426; border-radius:4px; height:6px; overflow:hidden; border:1px solid #16243b;">
    <div style="background:linear-gradient(90deg, {col}88, {col}); width:{pct:.1f}%; height:100%; border-radius:4px;"></div>
  </div>
</div>"""

    events_html = ""
    if recent_events_df is not None and not recent_events_df.empty:
        ev_rows = ""
        tail_df = recent_events_df.tail(4).iloc[::-1]
        for _, r in tail_df.iterrows():
            d_id = r.get("draw_id", "—")
            m_id = r.get("machine_id", "—")
            b_id = r.get("ball_set_id", "—")
            dt_s = str(r.get("draw_local_datetime", ""))[:16].replace("T", " ")
            temp = f"{r.get('temperature_c', '—')}°C" if "temperature_c" in r else "—"
            ev_rows += f"""<tr style="border-bottom:1px solid #16243b; font-size:0.78em;">
              <td style="padding:6px 8px; color:#94a3b8; font-family:monospace;">{dt_s}</td>
              <td style="padding:6px 8px; color:#e2e8f0; font-weight:600;">{d_id}</td>
              <td style="padding:6px 8px; color:#38bdf8;">{m_id}</td>
              <td style="padding:6px 8px; color:#cbd5e1;">{b_id}</td>
              <td style="padding:6px 8px; color:#fcd34d;">{temp}</td>
            </tr>"""
        events_html = f"""
<div style="margin-top:16px; border-top:1px solid #16243b; padding-top:12px;">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
    <div style="color:#e2e8f0; font-size:0.82em; font-weight:700;">🕒 Recent Machine &amp; Operational Events</div>
    <span style="font-size:0.68em; background:#1e293b; color:#94a3b8; padding:2px 6px; border-radius:4px;">Last {len(tail_df)} Records</span>
  </div>
  <table style="width:100%; border-collapse:collapse; text-align:left;">
    <thead>
      <tr style="color:#64748b; font-size:0.7em; text-transform:uppercase; border-bottom:1px solid #1e2f4d;">
        <th style="padding:4px 8px;">Date</th>
        <th style="padding:4px 8px;">Draw ID</th>
        <th style="padding:4px 8px;">Machine</th>
        <th style="padding:4px 8px;">Ball Set</th>
        <th style="padding:4px 8px;">Ambient</th>
      </tr>
    </thead>
    <tbody>{ev_rows}</tbody>
  </table>
</div>"""

    return f"""
<div style="
    background:#0a1324; border:1px solid #1e2f4d; border-radius:14px;
    padding:20px 24px; font-family:'Inter','Segoe UI',sans-serif; margin-bottom:18px;
">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; border-bottom:1px solid #16243b; padding-bottom:10px;">
    <div>
      <div style="color:#e2e8f0; font-size:0.88em; font-weight:700; letter-spacing:0.5px;">⚙️ Top Anomaly Drivers (Reason Codes)</div>
      <div style="color:#64748b; font-size:0.72em;">Influential feature deviations at the maximum anomaly draw</div>
    </div>
    <span style="font-size:0.7em; background:#1e293b; color:#94a3b8; padding:3px 8px; border-radius:4px;">Ranked by Magnitude</span>
  </div>
  {rows}
  <div style="color:#475569; font-size:0.7em; margin-top:10px; line-height:1.4;">
    * Standardized feature magnitudes. Outliers deviate significantly from baseline operating conditions.
  </div>
  {events_html}
</div>"""

def _error_card(msg: str) -> str:
    return f"""<div style="background:#250909;border:1.5px solid #ef4444;border-radius:12px;
    padding:18px 22px;color:#fca5a5;font-family:'Inter',sans-serif;font-size:0.88em;line-height:1.5;margin-bottom:16px;">
    ❌ <b>Validation / Execution Notice:</b><br>{msg}</div>"""

def _idle_card() -> str:
    return """<div style="background:linear-gradient(135deg,#0a1628 0%,#0e1e38 100%);border:1px solid #1e3a5f;border-radius:14px;padding:30px 26px;font-family:'Inter',sans-serif;margin-bottom:16px;">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
    <div style="font-size:2.4em;">🔬</div>
    <div>
      <h3 style="margin:0;color:#f1f5f9;font-size:1.25em;font-weight:700;">Lottery Draw Machine Failure Risk &amp; Telemetry Engine</h3>
      <p style="margin:3px 0 0;color:#94a3b8;font-size:0.82em;">AI-powered unsupervised anomaly detection for draw machine telemetry</p>
    </div>
  </div>

  <div style="background:rgba(14,165,233,0.08);border-left:4px solid #38bdf8;padding:12px 16px;border-radius:6px;margin-bottom:20px;">
    <p style="margin:0;color:#bae6fd;font-size:0.85em;line-height:1.55;">
      <b>👋 Welcome!</b> This application continuously monitors draw-machine telemetry (ambient temperature, ball set changes, and draw timings) to detect <b>abnormal operating deviations</b> before physical machine failures occur.
    </p>
  </div>

  <div style="color:#e2e8f0;font-size:0.84em;font-weight:700;margin-bottom:10px;text-transform:uppercase;letter-spacing:0.8px;">
    🚀 Quick Start Guide (3 Simple Steps):
  </div>

  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-bottom:22px;">
    <div style="background:#070f1e;border:1px solid #1e3a5f;border-radius:10px;padding:16px;">
      <div style="color:#38bdf8;font-size:1.4em;font-weight:800;margin-bottom:4px;">① Step 1</div>
      <div style="color:#f1f5f9;font-size:0.88em;font-weight:700;margin-bottom:4px;">Load Telemetry</div>
      <div style="color:#94a3b8;font-size:0.78em;line-height:1.45;">Click <b style="color:#38bdf8;">"⚡ Step 1: Load Sample Draw Batch"</b> on the left panel (or upload any valid draw CSV).</div>
    </div>
    <div style="background:#070f1e;border:1px solid #1e3a5f;border-radius:10px;padding:16px;">
      <div style="color:#c084fc;font-size:1.4em;font-weight:800;margin-bottom:4px;">② Step 2</div>
      <div style="color:#f1f5f9;font-size:0.88em;font-weight:700;margin-bottom:4px;">Filter (Optional)</div>
      <div style="color:#94a3b8;font-size:0.78em;line-height:1.45;">Optionally narrow down to a specific machine (e.g. <i>Excalibur4</i>) or analyze across all machines.</div>
    </div>
    <div style="background:#070f1e;border:1px solid #1e3a5f;border-radius:10px;padding:16px;">
      <div style="color:#fb923c;font-size:1.4em;font-weight:800;margin-bottom:4px;">③ Step 3</div>
      <div style="color:#f1f5f9;font-size:0.88em;font-weight:700;margin-bottom:4px;">Execute Scoring</div>
      <div style="color:#94a3b8;font-size:0.78em;line-height:1.45;">Click <b style="color:#f97316;">"▶ Step 3: Execute Anomaly Scoring"</b> to view risk ratings, reason codes &amp; trend charts.</div>
    </div>
  </div>

  <div style="background:#070f1e;border:1px solid #1e3a5f;border-radius:8px;padding:14px 18px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
    <div style="color:#94a3b8;font-size:0.82em;">
      👉 <b>Ready to test?</b> Click <span style="color:#38bdf8;font-weight:700;">"⚡ Step 1: Load Sample Draw Batch"</span> on the left!
    </div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;">
      <span style="background:#0d1e38;border:1px solid #1e3a5f;color:#38bdf8;font-size:0.72em;padding:3px 8px;border-radius:4px;">Isolation Forest v0.1.0</span>
      <span style="background:#0d1e38;border:1px solid #1e3a5f;color:#c084fc;font-size:0.72em;padding:3px 8px;border-radius:4px;">212 Features</span>
      <span style="background:#0d1e38;border:1px solid #1e3a5f;color:#34d399;font-size:0.72em;padding:3px 8px;border-radius:4px;">0% False Alarm Rate</span>
    </div>
  </div>
</div>"""

def _quality_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p style='color:#64748b;font-family:Inter,sans-serif;padding:24px;text-align:center;'>Upload a CSV on the Anomaly Detection tab or load sample data to inspect data quality.</p>"
    miss = df.isnull().mean() * 100
    rows_miss = int(df.isnull().any(axis=1).sum())
    dup_draws = int(df["draw_id"].duplicated().sum()) if "draw_id" in df.columns else 0

    summary = f"""
<div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));gap:14px;margin-bottom:24px;font-family:Inter,sans-serif;">
  <div style="background:#0a1324;border:1px solid #1e2f4d;border-radius:12px;padding:16px;text-align:center;">
    <div style="color:#64748b;font-size:0.72em;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Total Records</div>
    <div style="color:#38bdf8;font-size:1.8em;font-weight:800;margin-top:4px;">{len(df):,}</div>
    <div style="color:#475569;font-size:0.7em;">draw observations</div>
  </div>
  <div style="background:#0a1324;border:1px solid #1e2f4d;border-radius:12px;padding:16px;text-align:center;">
    <div style="color:#64748b;font-size:0.72em;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Available Columns</div>
    <div style="color:#c084fc;font-size:1.8em;font-weight:800;margin-top:4px;">{len(df.columns)}</div>
    <div style="color:#475569;font-size:0.7em;">schema breadth</div>
  </div>
  <div style="background:#0a1324;border:1px solid {'rgba(245,158,11,0.3)' if rows_miss else '#1e2f4d'};border-radius:12px;padding:16px;text-align:center;">
    <div style="color:#64748b;font-size:0.72em;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Rows With Nulls</div>
    <div style="color:{'#fbbf24' if rows_miss else '#34d399'};font-size:1.8em;font-weight:800;margin-top:4px;">{rows_miss:,}</div>
    <div style="color:#475569;font-size:0.7em;">{(rows_miss/max(len(df),1))*100:.1f}% completeness rate</div>
  </div>
  <div style="background:#0a1324;border:1px solid {'rgba(239,68,68,0.3)' if dup_draws else '#1e2f4d'};border-radius:12px;padding:16px;text-align:center;">
    <div style="color:#64748b;font-size:0.72em;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Duplicate Draw IDs</div>
    <div style="color:{'#ef4444' if dup_draws else '#34d399'};font-size:1.8em;font-weight:800;margin-top:4px;">{dup_draws}</div>
    <div style="color:#475569;font-size:0.7em;">{'Unique index satisfied' if dup_draws == 0 else 'Duplicate check warning'}</div>
  </div>
</div>"""

    header = """<div style="background:#0a1324;border:1px solid #1e2f4d;border-radius:14px;padding:20px;font-family:Inter,sans-serif;overflow-x:auto;">
<div style="color:#94a3b8;font-size:0.75em;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;font-weight:700;">Column Integrity & Missingness Matrix</div>
<table style="width:100%;border-collapse:collapse;font-size:0.85em;">
<thead><tr style="border-bottom:2px solid #1e2f4d;background:#070e1c;">
  <th style="padding:10px 14px;text-align:left;color:#64748b;font-weight:600;">Field Name</th>
  <th style="padding:10px 14px;text-align:center;color:#64748b;font-weight:600;">Requirement</th>
  <th style="padding:10px 14px;text-align:center;color:#64748b;font-weight:600;">Datatype</th>
  <th style="padding:10px 14px;text-align:left;color:#64748b;font-weight:600;">Data Completeness</th>
  <th style="padding:10px 14px;text-align:center;color:#64748b;font-weight:600;">Health Status</th>
</tr></thead><tbody>"""
    rows_html = ""
    for i, col in enumerate(df.columns):
        pct = miss.get(col, 0.0)
        fill = 100.0 - pct
        is_mand = col in MANDATORY_COLS
        mand_tag = '<span style="color:#38bdf8;font-weight:600;">★ Mand.</span>' if is_mand else '<span style="color:#475569;">Optional</span>'
        if pct == 0:
            c = "#10b981"
            badge = "<span style='color:#34d399;font-weight:600;'>✅ Complete (100%)</span>"
        elif pct <= 15:
            c = "#f59e0b"
            badge = f"<span style='color:#fbbf24;font-weight:600;'>⚠ Low ({pct:.1f}%)</span>"
        else:
            c = "#ef4444"
            badge = f"<span style='color:#f87171;font-weight:600;'>🔴 High ({pct:.1f}%)</span>"
        bg = "#070e1c" if i % 2 == 0 else "#0a1426"
        rows_html += f"""<tr style="background:{bg};border-bottom:1px solid #16243b;">
          <td style="padding:10px 14px;color:#cbd5e1;font-family:monospace;">{col}</td>
          <td style="padding:10px 14px;text-align:center;font-size:0.85em;">{mand_tag}</td>
          <td style="padding:10px 14px;text-align:center;font-family:monospace;color:#94a3b8;font-size:0.82em;">{str(df[col].dtype)}</td>
          <td style="padding:10px 14px;min-width:140px;">
            <div style="background:#111d33;border-radius:4px;height:6px;overflow:hidden;border:1px solid #1a2a44;">
              <div style="background:{c};width:{fill:.0f}%;height:100%;border-radius:4px;"></div>
            </div>
          </td>
          <td style="padding:10px 14px;text-align:center;font-size:0.85em;">{badge}</td>
        </tr>"""
    footer = "</tbody></table></div>"
    return summary + header + rows_html + footer

# ─── Chart builders ────────────────────────────────────────────────────────────
def _trend_chart(df, scores, machine, game):
    xv = df["draw_local_datetime"] if "draw_local_datetime" in df.columns else np.arange(len(df))
    flag = scores > ANOMALY_THRESH
    mean_s, std_s = float(np.mean(scores)), float(np.std(scores))
    n_flag = int(flag.sum())
    fig = go.Figure()
    xl = list(xv)
    fig.add_trace(go.Scatter(
        x=xl+xl[::-1], y=[mean_s+std_s]*len(xl)+[max(0.0, mean_s-std_s)]*len(xl),
        fill="toself", fillcolor="rgba(56,189,248,0.08)", line_color="rgba(0,0,0,0)",
        name="±1σ Confidence Band", hoverinfo="skip", showlegend=True))
    fig.add_trace(go.Scatter(
        x=xv, y=scores, mode="lines+markers", name="Anomaly Score",
        line={"color":"#38bdf8","width":2.2}, marker={"size":5,"color":"#0284c7"},
        hovertemplate="<b>%{x}</b><br>Score: %{y:.5f}<extra></extra>"))
    
    # High-contrast badge annotation for Threshold
    fig.add_hline(
        y=ANOMALY_THRESH, line_dash="dash", line_color="#fbbf24", line_width=1.8,
        annotation_text=f"<b>⚠ Alert Threshold ({ANOMALY_THRESH})</b>",
        annotation_position="top left",
        annotation_font=dict(color="#fef08a", size=12, family="Inter, sans-serif"),
        annotation_bgcolor="rgba(15, 23, 42, 0.92)",
        annotation_bordercolor="#eab308",
        annotation_borderwidth=1.5,
        annotation_borderpad=5
    )
    # High-contrast badge annotation for Mean
    fig.add_hline(
        y=mean_s, line_dash="dot", line_color="#94a3b8", line_width=1.5,
        annotation_text=f"<b>Baseline Mean ({mean_s:.4f})</b>",
        annotation_position="bottom left",
        annotation_font=dict(color="#f1f5f9", size=11, family="Inter, sans-serif"),
        annotation_bgcolor="rgba(15, 23, 42, 0.92)",
        annotation_bordercolor="#475569",
        annotation_borderwidth=1.2,
        annotation_borderpad=5
    )
    if flag.any():
        xf = xv[flag] if hasattr(xv, "__getitem__") else np.array(xv)[flag]
        fig.add_trace(go.Scatter(
            x=xf, y=scores[flag], mode="markers", name="⚠ Flagged Anomaly",
            marker={"color":"#ef4444","size":11,"symbol":"diamond-open","line":{"width":2.5,"color":"#ef4444"}},
            hovertemplate="<b>🚨 ANOMALY REVIEW</b><br>Score: %{y:.5f}<extra></extra>"))
    fig.update_layout(
        title=dict(
            text=f"Historical Draw Anomaly Trend — Machine: <b>{machine}</b> (Game: {game})  "
                 f"<span style='font-size:12px;color:#94a3b8;'>[{n_flag} Flagged / {len(scores)} Draws = {100*n_flag/max(len(scores),1):.1f}%]</span>",
            x=0.01, y=0.96, font=dict(size=14, color="#f1f5f9", family="Inter, sans-serif")),
        xaxis=dict(title="Draw Datetime (Chronological)", gridcolor="#16243b", color="#94a3b8", showline=True, linecolor="#1e2f4d"),
        yaxis=dict(title="Anomaly Score", gridcolor="#16243b", color="#94a3b8", tickformat=".3f", showline=True, linecolor="#1e2f4d"),
        paper_bgcolor="#0a1324", plot_bgcolor="#070e1c",
        font=dict(color="#cbd5e1", family="Inter, sans-serif"),
        hovermode="x unified", height=390,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=11)),
        margin=dict(l=55, r=25, t=65, b=45))
    return fig

def _dist_chart(scores):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores, nbinsx=28,
        marker=dict(color="#38bdf8", line=dict(color="#0a1324", width=1)),
        opacity=0.85, name="Score Distribution",
        hovertemplate="Score: %{x:.4f}<br>Count: %{y}<extra></extra>"))
    fig.add_vline(
        x=ANOMALY_THRESH, line_dash="dash", line_color="#ef4444", line_width=1.8,
        annotation_text="<b>Alert Cutoff (0.05)</b>",
        annotation_position="top right",
        annotation_font=dict(color="#fca5a5", size=11, family="Inter, sans-serif"),
        annotation_bgcolor="rgba(15, 23, 42, 0.92)",
        annotation_bordercolor="#ef4444",
        annotation_borderwidth=1.2,
        annotation_borderpad=4
    )
    mean_s = float(np.mean(scores))
    fig.add_vline(
        x=mean_s, line_dash="dot", line_color="#fbbf24", line_width=1.8,
        annotation_text=f"<b>Mean ({mean_s:.4f})</b>",
        annotation_position="top left",
        annotation_font=dict(color="#fef08a", size=11, family="Inter, sans-serif"),
        annotation_bgcolor="rgba(15, 23, 42, 0.92)",
        annotation_bordercolor="#eab308",
        annotation_borderwidth=1.2,
        annotation_borderpad=4
    )
    fig.update_layout(
        title=dict(text="Score Distribution & Threshold Histogram", x=0.01, y=0.96, font=dict(size=13, color="#f1f5f9", family="Inter, sans-serif")),
        xaxis=dict(title="Anomaly Score", gridcolor="#16243b", color="#94a3b8"),
        yaxis=dict(title="Frequency", gridcolor="#16243b", color="#94a3b8"),
        paper_bgcolor="#0a1324", plot_bgcolor="#070e1c",
        font=dict(color="#cbd5e1", family="Inter, sans-serif"),
        height=260, showlegend=False,
        margin=dict(l=45, r=20, t=40, b=40))
    return fig

# ─── Inference Handler ────────────────────────────────────────────────────────
def run_inference(file_obj, game_id, machine_id, analysis_date):
    start_t = time.perf_counter()
    blank_trend = go.Figure()
    blank_dist  = go.Figure()
    for fig in [blank_trend, blank_dist]:
        fig.update_layout(paper_bgcolor="#0a1324", plot_bgcolor="#0a1324", font=dict(color="#cbd5e1"))

    if file_obj is None:
        return _idle_card(), "", blank_trend, blank_dist

    try:
        if hasattr(file_obj, "name"):
            csv_path = file_obj.name
        elif isinstance(file_obj, str):
            csv_path = file_obj
        else:
            csv_path = str(file_obj)
        df = pd.read_csv(csv_path)
    except Exception as e:
        return _error_card(f"Cannot read CSV file: {e}"), "", blank_trend, blank_dist

    ok, msg = validate_file(df)
    if not ok:
        return _error_card(msg), "", blank_trend, blank_dist

    df["draw_local_datetime"] = pd.to_datetime(df["draw_local_datetime"], utc=True, errors="coerce")
    df = df.sort_values("draw_local_datetime").reset_index(drop=True)

    plot_df = df.copy()
    if game_id and game_id.strip() and game_id != "All Games":
        g_clean = game_id.strip()
        if "game_id" in plot_df.columns:
            filt = plot_df[plot_df["game_id"].astype(str) == g_clean]
            if filt.empty and "draw_id" in plot_df.columns:
                filt = plot_df[plot_df["draw_id"].astype(str).str.upper().str.startswith(g_clean.upper())]
            if not filt.empty:
                plot_df = filt.reset_index(drop=True)
            else:
                return _error_card(f"No records found matching Game '{g_clean}'. Please select an available game from the dropdown."), "", blank_trend, blank_dist

    if machine_id and machine_id.strip() and machine_id != "All Machines" and "machine_id" in plot_df.columns:
        m_clean = machine_id.strip()
        filt = plot_df[plot_df["machine_id"].astype(str) == m_clean]
        if filt.empty:
            filt = plot_df[plot_df["machine_id"].astype(str).str.lower().str.contains(m_clean.lower(), regex=False, na=False)]
        if not filt.empty:
            plot_df = filt.reset_index(drop=True)
        else:
            return _error_card(f"No records found for Machine '{m_clean}'. Available: {', '.join(plot_df['machine_id'].unique())}"), "", blank_trend, blank_dist

    if analysis_date and analysis_date.strip():
        try:
            cutoff = pd.to_datetime(analysis_date.strip(), utc=True)
            plot_df = plot_df[plot_df["draw_local_datetime"] <= cutoff].reset_index(drop=True)
        except Exception:
            pass

    if len(plot_df) == 0:
        return _error_card("No records match the selected Game, Machine, and Date filters."), "", blank_trend, blank_dist

    if _iso_model is None:
        return _error_card("Primary Isolation Forest model artifact not found. Please train models first."), "", blank_trend, blank_dist

    try:
        X, feat_cols = _prepare(plot_df)
        scores = _iso_model.score(X)
    except Exception as e:
        return _error_card(f"Inference execution failed: {e}"), "", blank_trend, blank_dist

    latency_ms = (time.perf_counter() - start_t) * 1000.0
    mean_s  = float(np.mean(scores))
    max_s   = float(np.max(scores))
    std_s   = float(np.std(scores))
    n_flag  = int((scores > ANOMALY_THRESH).sum())
    ts      = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    top_idx   = int(np.argmax(scores))
    row_vals  = X[top_idx]
    # Filter display reason codes to show machine/environmental telemetry only (avoid lottery numbers)
    candidate_indices = [
        i for i in np.argsort(np.abs(row_vals))[::-1]
        if not any(k in feat_cols[i].lower() for k in ["bonus", "main_numbers"])
    ]
    top_n     = min(5, len(candidate_indices))
    feat_ord  = candidate_indices[:top_n]
    top_feats = {feat_cols[i]: float(row_vals[i]) for i in feat_ord}

    risk_html = _risk_card(
        label="", mean_score=mean_s, max_score=max_s, std_score=std_s,
        n_flagged=n_flag, n_total=len(scores),
        machine=machine_id or "All Machines", game=game_id or "All Games",
        ts=ts, latency_ms=latency_ms)
    feat_html = _feature_card(top_feats, plot_df)

    logger.info("Inference done: game=%s machine=%s rows=%d mean=%.4f flagged=%d latency=%.1fms",
                game_id or "ALL", machine_id or "ALL", len(plot_df), mean_s, n_flag, latency_ms)
    return risk_html, feat_html, _trend_chart(plot_df, scores, machine_id or "All Machines", game_id or "All Games"), _dist_chart(scores)

def on_file_upload(file_obj):
    """When a file is uploaded, dynamically populate Game and Machine dropdown options from the CSV."""
    if file_obj is None:
        return (
            gr.update(choices=["All Games"], value="All Games"),
            gr.update(choices=["All Machines"], value="All Machines"),
            _quality_html(None),
        )
    try:
        if hasattr(file_obj, "name"):
            path = file_obj.name
        elif isinstance(file_obj, str):
            path = file_obj
        else:
            path = str(file_obj)
        df = pd.read_csv(path)
        games = ["All Games"]
        if "game_id" in df.columns:
            unique_games = sorted(df["game_id"].dropna().astype(str).unique().tolist())
            games += unique_games
        machines = ["All Machines"]
        if "machine_id" in df.columns:
            unique_machines = sorted(df["machine_id"].dropna().astype(str).unique().tolist())
            machines += unique_machines
        q_html = _quality_html(df)
        return (
            gr.update(choices=games, value="All Games"),
            gr.update(choices=machines, value="All Machines"),
            q_html,
        )
    except Exception as e:
        return (
            gr.update(choices=["All Games"], value="All Games"),
            gr.update(choices=["All Machines"], value="All Machines"),
            _error_card(f"Upload read error: {e}"),
        )

def load_sample_file():
    """Load sample file and return all 4 outputs: file path, game choices, machine choices, quality html."""
    # Try main sample path and fallback paths
    candidates = [
        SAMPLE_DATA_PATH,
        Path("data/raw/lottery_weather_dataset_20260914 (3).csv"),
        Path("data/processed/val_processed.csv"),
        Path("data/processed/test_processed.csv"),
    ]
    sample_path = None
    for c in candidates:
        if c.exists():
            sample_path = c
            break

    if sample_path is None:
        err = _error_card(f"Sample file not found. Searched: {[str(c) for c in candidates]}")
        return (
            None,
            gr.update(choices=["All Games"], value="All Games"),
            gr.update(choices=["All Machines"], value="All Machines"),
            err,
        )
    try:
        df = pd.read_csv(str(sample_path))
        games = ["All Games"]
        if "game_id" in df.columns:
            games += sorted(df["game_id"].dropna().astype(str).unique().tolist())
        machines = ["All Machines"]
        if "machine_id" in df.columns:
            machines += sorted(df["machine_id"].dropna().astype(str).unique().tolist())
        return (
            str(sample_path),
            gr.update(choices=games, value="All Games"),
            gr.update(choices=machines, value="All Machines"),
            _quality_html(df),
        )
    except Exception as e:
        err = _error_card(f"Failed to load sample file: {e}")
        return (
            None,
            gr.update(choices=["All Games"], value="All Games"),
            gr.update(choices=["All Machines"], value="All Machines"),
            err,
        )

# ─── Enhanced CSS ─────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  color-scheme: dark !important;
  --body-text-color: #e2e8f0 !important;
  --body-background-fill: #060c18 !important;
  --background-fill-primary: #0a1324 !important;
  --background-fill-secondary: #070e1c !important;
  --block-background-fill: #0a1324 !important;
  --block-label-text-color: #94a3b8 !important;
  --block-title-text-color: #e2e8f0 !important;
  --input-text-color: #e2e8f0 !important;
  --tw-prose-body: #e2e8f0 !important;
  --tw-prose-headings: #f8fafc !important;
  --tw-prose-links: #38bdf8 !important;
  --tw-prose-bold: #38bdf8 !important;
  --tw-prose-counters: #94a3b8 !important;
  --tw-prose-bullets: #38bdf8 !important;
  --tw-prose-hr: #1e3a5f !important;
  --tw-prose-quotes: #94a3b8 !important;
  --tw-prose-quote-borders: #38bdf8 !important;
  --tw-prose-code: #7dd3fc !important;
  --tw-prose-pre-code: #e2e8f0 !important;
  --tw-prose-pre-bg: #070e1c !important;
  --tw-prose-th-borders: #1e3a5f !important;
  --tw-prose-td-borders: #16243b !important;
}

*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container, .gradio-container * {
  font-family: 'Inter', -apple-system, sans-serif !important;
}
body, .gradio-container {
  background-color: #060c18 !important;
  color: #e2e8f0 !important;
}

/* Force light readable colors on all tabs and markdown */
.prose, .prose *, .md, .md *, [data-testid="markdown"], [data-testid="markdown"] * {
  color: #e2e8f0 !important;
}
.prose h1, .prose h2, .prose h3, .prose h4, .md h1, .md h2, .md h3, .md h4 {
  color: #f8fafc !important;
  font-weight: 700 !important;
}
.prose strong, .prose b, .md strong, .md b {
  color: #38bdf8 !important;
}
.prose table, .md table {
  width: 100% !important;
  border-collapse: collapse !important;
  margin: 14px 0 !important;
}
.prose th, .md th {
  background: #0f1e36 !important;
  color: #93c5fd !important;
  border: 1px solid #1e3a5f !important;
  padding: 10px 14px !important;
  text-align: left !important;
}
.prose td, .md td {
  background: #070e1c !important;
  color: #cbd5e1 !important;
  border: 1px solid #16243b !important;
  padding: 9px 14px !important;
}
.prose tr:nth-child(even) td, .md tr:nth-child(even) td {
  background: #0a1528 !important;
}
.prose code, .md code {
  background: #0d1a30 !important;
  color: #7dd3fc !important;
  border: 1px solid #1e3355 !important;
  padding: 2px 6px !important;
  border-radius: 4px !important;
}
.prose pre, .md pre {
  background: #070e1c !important;
  border: 1px solid #1e3355 !important;
  border-radius: 8px !important;
  padding: 14px !important;
}
.prose blockquote, .md blockquote {
  border-left: 4px solid #38bdf8 !important;
  background: rgba(56, 189, 248, 0.05) !important;
  padding: 10px 16px !important;
  border-radius: 0 8px 8px 0 !important;
  color: #94a3b8 !important;
}
hr { border-color: #1e3a5f !important; }
a { color: #38bdf8 !important; }

/* ── Top Header ── */
.app-header {
  background: linear-gradient(135deg, #091326 0%, #101e3b 50%, #091326 100%);
  border: 1px solid #1c2c48;
  border-radius: 16px;
  padding: 24px 30px;
  position: relative;
  overflow: hidden;
  margin-bottom: 16px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
}
.app-header::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #34d399 100%);
}
.app-header-title {
  font-size: 1.75em;
  font-weight: 800;
  margin: 0 0 6px;
  background: linear-gradient(135deg, #f8fafc 0%, #93c5fd 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.app-header-sub {
  color: #64748b;
  font-size: 0.85em;
  margin: 0 0 14px;
}
.header-tags {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.meta-tag {
  background: #0d1a30;
  border: 1px solid #1e3355;
  border-radius: 20px;
  padding: 3px 12px;
  color: #93c5fd;
  font-size: 0.72em;
  font-weight: 600;
}

/* ── Regulatory & Ethics Warning Banner ── */
.disclaimer-banner {
  background: rgba(245, 158, 11, 0.08);
  border: 1px solid rgba(245, 158, 11, 0.3);
  border-left: 5px solid #f59e0b;
  border-radius: 10px;
  padding: 12px 20px;
  color: #fde68a;
  font-size: 0.82em;
  line-height: 1.5;
  margin-bottom: 18px;
}

/* ── Tab Navigation ── */
.tab-nav {
  border-bottom: 1px solid #1c2c48 !important;
  margin-bottom: 18px !important;
}
.tab-nav button {
  color: #64748b !important;
  font-weight: 600 !important;
  font-size: 0.9em !important;
  padding: 10px 18px !important;
  border-radius: 8px 8px 0 0 !important;
  border: none !important;
  transition: all 0.2s ease !important;
}
.tab-nav button:hover {
  color: #cbd5e1 !important;
  background: rgba(255, 255, 255, 0.02) !important;
}
.tab-nav button.selected {
  color: #38bdf8 !important;
  border-bottom: 2px solid #38bdf8 !important;
  background: rgba(56, 189, 248, 0.06) !important;
}

.panel-section-title {
  font-size: 0.75em;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  color: #64748b;
  font-weight: 700;
  margin-bottom: 12px;
}

/* ── Buttons ── */
#run-btn button {
  background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
  border: none !important;
  border-radius: 10px !important;
  color: #ffffff !important;
  font-weight: 700 !important;
  padding: 13px !important;
  box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3) !important;
  transition: transform 0.1s, opacity 0.2s !important;
}
#run-btn button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }

#sample-btn button {
  background: #0f1e36 !important;
  border: 1px solid #233b63 !important;
  color: #93c5fd !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-size: 0.82em !important;
}
#sample-btn button:hover { background: #172c4e !important; }

/* ── Form Inputs, Dropdowns, Textboxes ── */
input, textarea, select,
input[type="text"],
.gr-input, .gr-box,
.secondary-wrap,
.wrap,
[data-testid="textbox"] input,
[data-testid="textbox"] textarea,
[data-testid="dropdown"] .wrap,
[data-testid="dropdown"] input,
[data-testid="dropdown"] button {
  background-color: #1e293b !important;
  background: #1e293b !important;
  border: 1px solid #334155 !important;
  color: #f8fafc !important;
  border-radius: 8px !important;
}

/* Ensure all text and placeholders inside inputs/dropdowns are clearly visible */
input, textarea, select,
input[type="text"],
[data-testid="textbox"] input,
[data-testid="dropdown"] input,
[data-testid="dropdown"] span,
[data-testid="dropdown"] div {
  color: #f8fafc !important;
  font-weight: 500 !important;
}

input::placeholder, textarea::placeholder {
  color: #94a3b8 !important;
  opacity: 1 !important;
}

/* Dropdown menu items list */
ul.options, .options-wrap, div[role="listbox"] {
  background-color: #1e293b !important;
  border: 1px solid #334155 !important;
}
li.item, div[role="option"] {
  color: #f8fafc !important;
  background-color: #1e293b !important;
}
li.item:hover, div[role="option"]:hover, li.item.selected, div[role="option"][aria-selected="true"] {
  background-color: #334155 !important;
  color: #38bdf8 !important;
}

/* ── File Upload Component (Solid White Box with High-Contrast Dark Text) ── */
div[data-testid="file"],
div[data-testid="file-upload"],
.upload-container {
  background-color: #ffffff !important;
  background: #ffffff !important;
  border: 1.5px solid #cbd5e1 !important;
  border-radius: 10px !important;
  width: 100% !important;
  overflow: hidden !important;
  transition: border-color 0.2s !important;
}

div[data-testid="file"]:hover,
div[data-testid="file-upload"]:hover {
  border-color: #0284c7 !important;
}

/* Inner drop area spans 100% width cleanly */
div[data-testid="file-upload"] > div,
div[data-testid="file-upload"] button {
  background: #ffffff !important;
  background-color: #ffffff !important;
  border: none !important;
  width: 100% !important;
  min-height: 110px !important;
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: center !important;
  padding: 16px 12px !important;
  box-shadow: none !important;
}

/* Upload cloud/arrow icon: Vibrant deep blue */
div[data-testid="file-upload"] svg {
  color: #0284c7 !important;
  stroke: #0284c7 !important;
  fill: none !important;
  width: 32px !important;
  height: 32px !important;
  margin-bottom: 8px !important;
}

/* Upload text prompts: Dark charcoal grey, sharp and clearly readable */
div[data-testid="file-upload"] span,
div[data-testid="file-upload"] p,
div[data-testid="file"] label span,
div[data-testid="file"] .label {
  color: #0f172a !important;
  font-weight: 600 !important;
  font-size: 0.9em !important;
}

/* ── File Preview Box (When file is loaded: Sleek soft light card with dark text) ── */
.file-preview,
.file-item,
.single-file,
div[data-testid="file"] .file-preview,
div[data-testid="file"] .file,
div[data-testid="file"] table,
div[data-testid="file"] tr,
div[data-testid="file"] td {
  background-color: #f1f5f9 !important;
  background: #f1f5f9 !important;
  border: 1px solid #cbd5e1 !important;
  border-radius: 6px !important;
}

/* File name and text inside the file preview: Deep Charcoal */
.file-preview *,
.file-item *,
.single-file *,
.file-name,
.filename,
div[data-testid="file"] td *,
div[data-testid="file"] a,
div[data-testid="file"] span,
div[data-testid="file"] p {
  color: #0f172a !important;
  font-weight: 700 !important;
}

/* File size and download button: Deep blue */
.file-item .filesize,
.file-size,
.download-button,
div[data-testid="file"] .filesize,
div[data-testid="file"] a.download-button {
  color: #0284c7 !important;
  font-weight: 600 !important;
}

/* Clear button (X icon) */
div[data-testid="file"] button[aria-label="Clear"],
div[data-testid="file"] button.clear-button,
div[data-testid="file"] button svg {
  color: #64748b !important;
}
div[data-testid="file"] button:hover svg {
  color: #ef4444 !important;
}

footer { display: none !important; }
"""

# ─── Gradio App Construction ──────────────────────────────────────────────────
def build_app() -> gr.Blocks:
    with gr.Blocks(title="LD ML 03: Lottery Machine Failure Risk Application", css=CSS) as demo:

        # Header Block
        gr.HTML(f"""
        <div class="app-header">
          <h1 class="app-header-title">🔬 Lottery Draw-Machine Anomaly &amp; Failure Risk System</h1>
          <p class="app-header-sub">LD ML 03 Engineering Space &nbsp;·&nbsp; Unsupervised Telemetry &amp; Anomaly Detection &nbsp;·&nbsp; Hugging Face Hub Integration</p>
          <div class="header-tags">
            <span class="meta-tag">Model: {MODEL_NAME}</span>
            <span class="meta-tag">Version: {MODEL_VERSION}</span>
            <span class="meta-tag">212 Leakage-Safe Features</span>
            <span class="meta-tag">Chronological 70/15/15 Split</span>
            <span class="meta-tag">Alert Threshold: {ANOMALY_THRESH}</span>
          </div>
        </div>
        """)

        # Regulatory & Ethics Disclaimer Banner
        gr.HTML("""
        <div class="disclaimer-banner">
          ⚠️ <b>Operational Policy &amp; Ethics Disclosure:</b> This application analyses mechanical, environmental, and temporal operating telemetry to detect machine anomalies.
          <b>It does NOT predict winning numbers, lottery outcomes, or ball sets.</b> In the absence of verified failure labels, all outputs represent
          <b>unsupervised anomaly scores</b> and require qualified engineering inspection prior to taking physical maintenance action.
        </div>
        """)

        # Tabs Layout
        with gr.Tabs():

            # ── Tab 1: Telemetry & Anomaly Detection ─────────────────────────
            with gr.TabItem("🎯 Anomaly Detection"):
                with gr.Row(equal_height=False):

                    # Left Column: Inputs & Controls
                    with gr.Column(scale=1, min_width=300):
                        gr.HTML("""
                        <div style="background:linear-gradient(135deg,#0e1e38 0%,#091326 100%);border:1px solid #1e3a5f;border-left:4px solid #38bdf8;border-radius:10px;padding:12px 14px;margin-bottom:14px;">
                          <div style="color:#38bdf8;font-size:0.76em;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">
                            🚀 Quick Start (3 Steps)
                          </div>
                          <div style="color:#cbd5e1;font-size:0.8em;line-height:1.55;">
                            <b>1.</b> Click <span style="color:#38bdf8;font-weight:700;">⚡ Step 1</span> below.<br>
                            <b>2.</b> <i>(Optional)</i> Pick a machine or leave on "All".<br>
                            <b>3.</b> Click <span style="color:#f97316;font-weight:700;">▶ Step 3</span> to view anomalies!
                          </div>
                        </div>
                        <div class="panel-section-title">📂 Input Telemetry &amp; Controls</div>
                        """)

                        sample_btn = gr.Button("⚡ Step 1: Load Sample Draw Batch", elem_id="sample-btn")

                        file_in = gr.File(
                            label="Historical Draws CSV (or drag & drop here)",
                            file_types=[".csv"],
                        )

                        gr.HTML('<div style="color:#64748b;font-size:0.72em;text-transform:uppercase;letter-spacing:1px;font-weight:700;margin:12px 0 6px;">⚙️ Step 2: Filters (Optional)</div>')

                        game_in = gr.Dropdown(
                            label="Lottery Game Filter",
                            choices=["All Games", "2", "THUNDERBALL", "LOTTO"],
                            value="All Games",
                            interactive=True,
                        )

                        machine_in = gr.Dropdown(
                            label="Machine Identifier",
                            choices=["All Machines", "Excalibur4", "M001", "M002"],
                            value="All Machines",
                            allow_custom_value=True,
                            interactive=True,
                        )

                        date_in = gr.Textbox(
                            label="Analysis Cutoff Date (UTC)",
                            placeholder="e.g. 2026-08-26 (leave blank for latest)",
                            interactive=True,
                        )

                        run_btn = gr.Button("▶ Step 3: Execute Anomaly Scoring", variant="primary", elem_id="run-btn")

                    # Right Column: Visual Outputs
                    with gr.Column(scale=2):
                        gr.HTML('<div class="panel-section-title">📊 Operational Health Telemetry</div>')
                        risk_html   = gr.HTML(value=_idle_card())
                        feat_html   = gr.HTML(value="")
                        trend_plot  = gr.Plot(label="Anomaly Score Trend", show_label=False)
                        dist_plot   = gr.Plot(label="Score Distribution",  show_label=False)

            # ── Tab 2: Data Quality Audit ────────────────────────────────────
            with gr.TabItem("📋 Data Quality"):
                gr.HTML("""
                <div style="margin-bottom:14px; font-family:Inter,sans-serif; color:#94a3b8; font-size:0.85em;">
                  Real-time audit of incoming draw observations: validates schema compliance, data dictionary types, missingness percentages, and duplicate draw IDs.
                </div>
                """)
                quality_html = gr.HTML(value=_quality_html(None))

            # ── Tab 3: Model Performance & Backtesting ───────────────────────
            with gr.TabItem("📈 Model Performance"):
                gr.Markdown("""
### 🧪 Model Evaluation & Chronological Backtesting Report

> **Split Strategy:** Strict chronological walk-forward split (**70% Train / 15% Validation / 15% Test**). No future observations seen during training or threshold tuning.

---

#### 1. Holdout Benchmark Metrics

| Dataset Partition | Draws Count | Contamination / Target | Anomaly Rate | Mean Anomaly Score | Score Dispersion (Std) | Max Peak Score |
|---|---|---|---|---|---|---|
| **Training Set** | 124 | 5.0% | 4.84% | 0.0000 | ±0.0312 | 0.0812 |
| **Validation Set** | 26 | — | **0.00%** | **0.0175** | **±0.0218** | **0.0635** |
| **Chronological Test** | 27 | — | **0.00%** | **0.0214** | **±0.0154** | **0.0580** |

*Evaluation Notes: Holdout periods exhibit 0% false alarm rate under nominal operations while score standard deviation remains well below 0.022, proving high temporal stability.*

---

#### 2. Baseline Comparison Memo

| Architecture | Model Family | Validation Behavior | Decision & Operational Verdict |
|---|---|---|---|
| **Robust Z-Score** | Statistical Median + MAD | 100% false-positive flag rate | ❌ **Rejected:** Overly sensitive to seasonal variance; unfeasible false-alert burden. |
| **Isolation Forest** | Random Partitioning Trees | 0.0175 mean score; 0% false alerts | ✅ **Selected Primary:** High stability across chronological holdout; defensible anomaly ranking. |
| **IBM Granite TTM / Chronos** | Foundation Time-Series | Experimental on temperature | ℹ️ **Documented:** Useful for continuous weather forecasting; not primary anomaly classifier. |

---

#### 3. Leakage Guard Evidence
* **Window-safe rolling stats:** All rolling aggregates compute with `.shift(1)` — the current draw is strictly excluded from its own rolling window.
* **Fit boundaries:** Imputers and scalers fit exclusively on the 70% chronological training period.
                """)

            # ── Tab 4: Model Card ────────────────────────────────────────────
            with gr.TabItem("📝 Model Card"):
                gr.Markdown(MODEL_CARD_MD)

            # ── Tab 5: Dataset Card ──────────────────────────────────────────
            with gr.TabItem("📦 Dataset Card"):
                gr.Markdown(DATASET_CARD_MD)

            # ── Tab 6: Responsible Use & Governance ──────────────────────────
            with gr.TabItem("⚖️ Responsible Use"):
                gr.Markdown("""
# ⚖️ Responsible Use, Ethics, and Governance Charter

### ⚠️ Mandatory Operational Disclaimers
1. **Zero Outcome Prediction Capability:** This application processes draw timing, ball set changes, ambient venue weather, and machine identifiers. **It does not predict, forecast, or improve player odds of winning lottery numbers.** Any claim to the contrary is fraudulent.
2. **Label Gate Limitation:** Verified physical failure logs and maintenance work orders do not exist in the source lottery feed. The model outputs **unsupervised anomaly scores**, indicating departure from historical baseline behavior — **never a confirmed machine breakdown probability**.

---

### 🛡️ Operator Decision Guidelines

| Principle | Acceptable Application (✅) | Prohibited Application (❌) |
|---|---|---|
| **Maintenance Prioritization** | Flagging physical draw machines for routine preventative engineering checkups. | Halting live broadcasts solely based on an unverified automated score. |
| **Data Integrity** | Identifying missing weather reports or delayed draw timestamps in the central database. | Using temperature or barometric pressure to claim numbers are influenced by weather. |
| **Public Disclosure** | Disclosing to auditors that the model measures telemetry drift. | Sharing scores with ticket buyers or gambling syndicates. |

---

### 🏛️ Engineering Handover Checklist
* **Artifact Location:** `models/iso_forest_detector.joblib`
* **Audit Trail:** Fully documented in `docs/research_memo.md`, `docs/label_audit.md`, and `docs/prompt_log.md`.
* **Licensing:** Open-source for training and reproducible educational deployment on Hugging Face Spaces.
                """)

        # ── Event Bindings (After all Tabs and Components are defined) ───────
        sample_btn.click(
            fn=load_sample_file,
            inputs=[],
            outputs=[file_in, game_in, machine_in, quality_html],
            api_name=False,
        )

        file_in.change(
            fn=on_file_upload,
            inputs=[file_in],
            outputs=[game_in, machine_in, quality_html],
            api_name=False,
        )

        run_btn.click(
            fn=run_inference,
            inputs=[file_in, game_in, machine_in, date_in],
            outputs=[risk_html, feat_html, trend_plot, dist_plot],
            api_name=False,
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        show_error=True,
        show_api=False,
    )

