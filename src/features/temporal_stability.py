"""
src/features/temporal_stability.py
------------------------------------
Temporal stability features: rolling z-scores, drift indicators,
and recent anomaly counts.

All rolling operations apply shift(1) before the window to prevent
look-ahead leakage. The current row's value is never included in its
own feature computation.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SKIP_COLS = {
    "game_id",
    "draw_id",
    "machine_id",
    "ball_set_id",
    "venue_id",
    "draw_local_datetime",
}


def compute_temporal_features(
    df: pd.DataFrame,
    window: int = 30,
) -> pd.DataFrame:
    """
    Compute temporal stability features for all numeric columns.

    For each eligible numeric column ``col``, the following features are added:

    - ``{col}_rolling_zscore``   : z-score of the current value relative to
                                   the rolling mean and std of the prior
                                   ``window`` rows (shift applied).
    - ``{col}_drift``            : difference between the current rolling mean
                                   and the rolling mean ``window`` steps ago,
                                   indicating trend direction.
    - ``{col}_recent_anomalies`` : count of rows in the rolling window where
                                   the absolute z-score exceeded 2.0 (a proxy
                                   for how frequently the signal was anomalous
                                   in the recent past).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Must be sorted chronologically.
    window : int, optional
        Rolling window size in rows. Defaults to 30.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with new temporal feature columns appended.
    """
    df = df.copy()

    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in _SKIP_COLS
    ]

    if not numeric_cols:
        logger.warning("No numeric columns found for temporal feature computation.")
        return df

    new_features = {}
    for col in numeric_cols:
        shifted = df[col].shift(1)
        rolling = shifted.rolling(window=window, min_periods=2)

        roll_mean = rolling.mean()
        roll_std = rolling.std().replace(0, np.nan)

        # Rolling z-score: how many stds is the current value from recent mean
        new_features[f"{col}_rolling_zscore"] = (df[col] - roll_mean) / roll_std

        # Drift: difference between current rolling mean and the mean `window`
        # steps ago (double rolling, shift already applied)
        older_mean = roll_mean.shift(window)
        new_features[f"{col}_drift"] = roll_mean - older_mean

        # Recent anomaly count: fraction of prior window where |zscore| > 2
        prior_zscore = ((shifted - roll_mean) / roll_std).abs()
        new_features[f"{col}_recent_anomalies"] = (
            prior_zscore.rolling(window=window, min_periods=1)
            .apply(lambda x: (x > 2.0).sum(), raw=True)
            .fillna(0)
            .astype(int)
        )

    if new_features:
        new_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, new_df], axis=1)

    logger.info(
        "Temporal stability features computed for %d column(s) with window=%d.",
        len(numeric_cols),
        window,
    )
    return df
