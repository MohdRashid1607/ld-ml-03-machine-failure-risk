"""
src/data/transformer.py
-----------------------
Data cleaning, categorical encoding, and numeric scaling utilities.

Fitted encoder/scaler objects are persisted to disk with joblib so that
the same transformations can be applied consistently to inference data.
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

_ARTIFACTS_DIR = Path("models/transformers")
_SCALER_PATH = _ARTIFACTS_DIR / "scaler.joblib"
_ENCODERS_PATH = _ARTIFACTS_DIR / "label_encoders.joblib"

# Columns to exclude from encoding/scaling (IDs, targets, datetime)
_PASSTHROUGH_COLS = {
    "game_id",
    "draw_id",
    "machine_id",
    "ball_set_id",
    "venue_id",
    "draw_local_datetime",
}


def _identify_column_types(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Return (categorical_cols, numeric_cols) excluding passthrough columns."""
    cat_cols = [
        c
        for c in df.select_dtypes(include=["object", "category"]).columns
        if c not in _PASSTHROUGH_COLS
    ]
    num_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in _PASSTHROUGH_COLS
    ]
    return cat_cols, num_cols


def clean_and_encode(
    df: pd.DataFrame,
    fit: bool = True,
) -> pd.DataFrame:
    """
    Clean a DataFrame, encode categorical features, and scale numeric features.

    When ``fit=True`` (training mode) the LabelEncoders and StandardScaler are
    fitted on the supplied data and persisted to ``models/transformers/`` using
    joblib.  When ``fit=False`` (inference mode) those persisted objects are
    loaded and applied without re-fitting.

    Processing steps
    ----------------
    1. Drop entirely-empty columns.
    2. Fill numeric NaNs with column median.
    3. Fill categorical NaNs with the placeholder string ``"__missing__"``.
    4. Label-encode each categorical column.
    5. Standard-scale each numeric column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (should already be validated).
    fit : bool, optional
        If True, fit and save transformers. If False, load and apply saved
        transformers. Defaults to True.

    Returns
    -------
    pd.DataFrame
        Transformed copy of the input DataFrame.  Passthrough columns and
        datetime columns are preserved unchanged.
    """
    df = df.copy()

    cat_cols, num_cols = _identify_column_types(df)

    # 1. Fill numeric NaNs with median
    for col in num_cols:
        median_val = df[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        filled = df[col].isna().sum()
        if filled > 0:
            df[col] = df[col].fillna(median_val)
            logger.debug("Filled %d NaNs in '%s' with median %.4f.", filled, col, median_val)

    # 3. Fill categorical NaNs
    for col in cat_cols:
        df[col] = df[col].fillna("__missing__").astype(str)

    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if fit:
        # 4. Fit & apply label encoders
        encoders: dict[str, Any] = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        joblib.dump(encoders, _ENCODERS_PATH)
        logger.info("Label encoders saved to %s", _ENCODERS_PATH)

        # 5. Fit & apply scaler
        if num_cols:
            scaler = StandardScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
            joblib.dump(scaler, _SCALER_PATH)
            logger.info("StandardScaler saved to %s", _SCALER_PATH)
    else:
        # Load and apply saved transformers
        if _ENCODERS_PATH.exists():
            encoders = joblib.load(_ENCODERS_PATH)
            for col in cat_cols:
                if col in encoders:
                    le = encoders[col]
                    # Handle unseen labels gracefully
                    known = set(le.classes_)
                    df[col] = df[col].apply(
                        lambda x, k=known, enc=le: enc.transform([x])[0]
                        if x in k
                        else -1
                    )
        else:
            logger.warning("No saved encoders found at %s. Skipping.", _ENCODERS_PATH)

        if num_cols and _SCALER_PATH.exists():
            scaler = joblib.load(_SCALER_PATH)
            df[num_cols] = scaler.transform(df[num_cols])
        elif num_cols:
            logger.warning("No saved scaler found at %s. Skipping.", _SCALER_PATH)

    return df
