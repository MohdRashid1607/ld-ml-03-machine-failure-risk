"""
src/features/data_quality.py
------------------------------
Data quality meta-features.

Computes per-row missingness ratio and adds a binary flag for rows
with high missingness — useful for downstream anomaly detection and
model diagnostics.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that are always expected to be present; exclude from missingness ratio
_EXCLUDE_COLS = {"draw_local_datetime"}

# Threshold above which a row is considered high-missingness
_DEFAULT_MISSINGNESS_THRESHOLD = 0.20


def compute_quality_features(
    df: pd.DataFrame,
    threshold: float = _DEFAULT_MISSINGNESS_THRESHOLD,
) -> pd.DataFrame:
    """
    Compute per-row data quality (missingness) features.

    Features added
    --------------
    missingness_ratio : float
        The fraction of feature columns in each row that are NaN.
        Range [0.0, 1.0].
    missingness_flag : int
        Binary flag (1 = high missingness, 0 = acceptable). Rows with
        ``missingness_ratio > threshold`` are flagged.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (any stage of the pipeline).
    threshold : float, optional
        The missingness ratio above which a row is flagged.
        Defaults to 0.20 (20 %).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with ``missingness_ratio`` and ``missingness_flag``
        columns appended.
    """
    df = df.copy()

    # Consider only columns that are not explicitly excluded
    feature_cols = [c for c in df.columns if c not in _EXCLUDE_COLS]

    if not feature_cols:
        logger.warning("No eligible columns found for missingness computation.")
        df["missingness_ratio"] = 0.0
        df["missingness_flag"] = 0
        return df

    df["missingness_ratio"] = df[feature_cols].isna().mean(axis=1)
    df["missingness_flag"] = (df["missingness_ratio"] > threshold).astype(int)

    flagged = df["missingness_flag"].sum()
    logger.info(
        "Quality features computed: %.1f%% of rows flagged for high missingness "
        "(threshold=%.0f%%).",
        100.0 * flagged / len(df),
        100.0 * threshold,
    )
    return df
