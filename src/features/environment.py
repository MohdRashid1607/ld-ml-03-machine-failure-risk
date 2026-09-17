"""
src/features/environment.py
----------------------------
Environmental and temporal context feature engineering.

Derives time-of-day, day-of-week, month, season, and rolling temperature
features from the ``draw_local_datetime`` and ``temperature_c`` columns.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_SEASON_MAP = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}


def _month_to_season(month: int) -> str:
    return _SEASON_MAP.get(month, "unknown")


def compute_environment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add environmental and calendar context features to a DataFrame.

    Features added
    --------------
    hour_of_day : int
        Hour extracted from ``draw_local_datetime`` (0–23).
    day_of_week : int
        Day of week (0 = Monday … 6 = Sunday).
    month : int
        Calendar month (1–12).
    season : str
        Season label derived from the month: winter / spring / summer / autumn.
    temperature_rolling_mean : float
        Rolling 7-draw mean of ``temperature_c``, using shift(1) to avoid
        leakage (the current row's temperature is not included in its own mean).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Must contain ``draw_local_datetime`` (datetime-like)
        and ``temperature_c`` (numeric).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with new environment feature columns appended.
    """
    df = df.copy()

    if "draw_local_datetime" not in df.columns:
        raise KeyError("'draw_local_datetime' column is required.")

    dt = df["draw_local_datetime"]

    df["hour_of_day"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["season"] = df["month"].map(_month_to_season)

    if "temperature_c" in df.columns:
        df["temperature_rolling_mean"] = (
            df["temperature_c"]
            .shift(1)
            .rolling(window=7, min_periods=1)
            .mean()
        )
    else:
        logger.warning(
            "'temperature_c' not found; 'temperature_rolling_mean' will not be added."
        )

    logger.info("Environment features computed: %d rows.", len(df))
    return df
