"""
src/features/distribution_diagnostics.py
-----------------------------------------
Rolling distribution diagnostic features for numeric columns.

All rolling computations use shift(1) before the rolling window to
ensure no look-ahead leakage — the current row's value is never
included in its own feature computation.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns to skip (IDs, datetime, non-numeric)
_SKIP_COLS = {
    "game_id",
    "draw_id",
    "machine_id",
    "ball_set_id",
    "venue_id",
    "draw_local_datetime",
}


def compute_distribution_features(
    df: pd.DataFrame,
    window: int = 30,
) -> pd.DataFrame:
    """
    Compute rolling distribution diagnostics for all numeric columns.

    For each eligible numeric column ``col``, the following features are added:

    - ``{col}_rolling_mean``      : rolling mean over ``window`` rows
    - ``{col}_rolling_var``       : rolling variance over ``window`` rows
    - ``{col}_rolling_sum``       : rolling sum over ``window`` rows
    - ``{col}_rolling_range``     : rolling max minus rolling min
    - ``{col}_rolling_freq_dev``  : deviation of the rolling mean from the
                                    global mean (absolute value)

    All rolling windows use ``shift(1)`` applied before rolling to ensure the
    current row is **not** included in its own feature computation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Should be sorted chronologically.
    window : int, optional
        Number of rows for the rolling window. Defaults to 30.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with new distribution feature columns appended.
    """
    df = df.copy()

    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in _SKIP_COLS
    ]

    if not numeric_cols:
        logger.warning("No numeric columns found for distribution feature computation.")
        return df

    for col in numeric_cols:
        # Shift by 1 to avoid leakage (current row excluded from its own window)
        shifted = df[col].shift(1)
        global_mean = df[col].mean()

        rolling = shifted.rolling(window=window, min_periods=1)

        df[f"{col}_rolling_mean"] = rolling.mean()
        df[f"{col}_rolling_var"] = rolling.var()
        df[f"{col}_rolling_sum"] = rolling.sum()
        df[f"{col}_rolling_range"] = rolling.max() - rolling.min()
        df[f"{col}_rolling_freq_dev"] = (rolling.mean() - global_mean).abs()

    logger.info(
        "Distribution features computed for %d column(s) with window=%d.",
        len(numeric_cols),
        window,
    )
    return df
