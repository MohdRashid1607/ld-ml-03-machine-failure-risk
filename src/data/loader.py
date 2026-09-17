"""
src/data/loader.py
------------------
Data loading utilities for the Machine Failure Risk project.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: list[str] = [
    "game_id",
    "draw_id",
    "machine_id",
    "ball_set_id",
    "venue_id",
    "draw_local_datetime",
    "temperature_c",
]

DATETIME_COLUMNS: list[str] = ["draw_local_datetime"]


def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV file, validate required columns, parse datetime columns,
    and return a clean DataFrame.

    Parameters
    ----------
    path : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        A cleaned DataFrame with properly parsed datetime columns and
        the index reset.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the provided path.
    ValueError
        If one or more required columns are missing from the CSV.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path.resolve()}")

    logger.info("Loading CSV from: %s", file_path.resolve())
    df = pd.read_csv(file_path, low_memory=False)

    # Validate required columns
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Parse datetime columns
    for col in DATETIME_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
            nat_count = df[col].isna().sum()
            if nat_count > 0:
                logger.warning(
                    "Column '%s' has %d unparsable datetime values (set to NaT).",
                    col,
                    nat_count,
                )

    # Sort chronologically by the primary datetime column
    if "draw_local_datetime" in df.columns:
        df = df.sort_values("draw_local_datetime").reset_index(drop=True)

    logger.info("CSV loaded successfully: %d rows, %d columns.", *df.shape)
    return df
