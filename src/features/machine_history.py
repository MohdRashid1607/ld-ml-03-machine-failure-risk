"""
src/features/machine_history.py
--------------------------------
Machine-level historical feature engineering.

⚠️  LEAKAGE WARNING: All features in this module are computed using only
information available at or before the current row's timestamp.  This is
enforced by using cumulative expanding/shift operations keyed on the
chronologically sorted draw sequence.  Never use future data (e.g. lead
windows) in this module.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_machine_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-machine historical usage features with strict temporal safety.

    All features are derived from information available at or before each
    row's ``draw_local_datetime``, using cumulative (expanding) windows and
    shift(1) offsets to prevent any look-ahead leakage.

    Features added
    --------------
    draw_count_since_install : int
        Cumulative count of draws for this machine up to (but not including)
        the current draw (shift(1) applied so the count reflects draws *before*
        the current row).
    time_since_last_service : float
        Days elapsed since the previous draw for this machine.  Computed as
        the difference between the current timestamp and the previous
        timestamp (shifted by 1).  NaN for the first draw of each machine.
    recent_machine_changes : int
        Rolling count of draws in the last 30 draws for this machine.
        Uses shift(1) before rolling to avoid seeing the current draw.
    cumulative_usage : int
        Total number of draws ever recorded for this machine up to the
        current row (inclusive expanding count, then shifted by 1).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Must be sorted chronologically by
        ``draw_local_datetime`` and contain ``machine_id``.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with the four new feature columns appended.
    """
    if "machine_id" not in df.columns:
        raise KeyError("'machine_id' column is required.")
    if "draw_local_datetime" not in df.columns:
        raise KeyError("'draw_local_datetime' column is required.")

    df = df.copy().sort_values("draw_local_datetime").reset_index(drop=True)

    grp = df.groupby("machine_id", sort=False)

    # --- draw_count_since_install ---
    # Cumulative count within group, shifted so current row sees previous count
    df["draw_count_since_install"] = (
        grp.cumcount()  # 0-indexed cumulative count (does NOT include current row implicitly)
    )
    # cumcount() already returns the count of prior rows in the group (0 for first),
    # so no additional shift needed.

    # --- cumulative_usage ---
    # Same as draw_count_since_install here (0-indexed prior draws)
    df["cumulative_usage"] = df["draw_count_since_install"]

    # --- time_since_last_service ---
    # Difference in days between current draw and the previous draw in the group
    df["_prev_datetime"] = grp["draw_local_datetime"].shift(1)
    df["time_since_last_service"] = (
        df["draw_local_datetime"] - df["_prev_datetime"]
    ).dt.total_seconds() / 86_400.0
    df.drop(columns=["_prev_datetime"], inplace=True)

    # --- recent_machine_changes ---
    # Rolling count of the last 30 draws; shift(1) prevents current-row inclusion
    def _rolling_count(series: pd.Series, window: int = 30) -> pd.Series:
        """Return rolling count over the last `window` draws, leakage-safe."""
        shifted = series.shift(1)
        return shifted.rolling(window=window, min_periods=1).count().fillna(0).astype(int)

    df["recent_machine_changes"] = grp["draw_id"].transform(
        lambda s: _rolling_count(s)
    )

    logger.info(
        "Machine history features computed: %d rows, columns added: "
        "draw_count_since_install, time_since_last_service, "
        "recent_machine_changes, cumulative_usage.",
        len(df),
    )
    return df
