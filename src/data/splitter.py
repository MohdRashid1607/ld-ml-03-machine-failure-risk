"""
src/data/splitter.py
--------------------
Chronological data splitting utilities for the Machine Failure Risk project.

Splits are performed strictly by date to prevent any temporal data leakage.
The data is never shuffled.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

DATETIME_COL = "draw_local_datetime"


def chronological_split(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into training, validation, and test sets by date.

    All rows with ``draw_local_datetime`` <= ``train_end`` go to train.
    Rows with dates in (``train_end``, ``val_end``] go to validation.
    Rows with dates after ``val_end`` go to test.

    The order of rows is preserved (no shuffling) to avoid temporal leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Must contain the ``draw_local_datetime`` column
        with datetime-like values (timezone-aware or naive are both accepted,
        but must be consistent).
    train_end : str
        Upper boundary for the training set, e.g. ``'2023-06-30'``.
        Parsed with ``pd.Timestamp``.
    val_end : str
        Upper boundary for the validation set, e.g. ``'2023-12-31'``.
        Parsed with ``pd.Timestamp``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A 3-tuple of ``(train_df, val_df, test_df)``, each with the original
        index reset.

    Raises
    ------
    KeyError
        If ``draw_local_datetime`` column is not present in ``df``.
    ValueError
        If ``train_end`` >= ``val_end``, or if any split is empty.
    """
    if DATETIME_COL not in df.columns:
        raise KeyError(
            f"Column '{DATETIME_COL}' not found. "
            "Ensure the DataFrame has been loaded via load_csv()."
        )

    t_end = pd.Timestamp(train_end)
    v_end = pd.Timestamp(val_end)

    if t_end >= v_end:
        raise ValueError(
            f"train_end ({train_end}) must be strictly before val_end ({val_end})."
        )

    # Normalise timezone: make boundary timestamps tz-aware if the column is
    dt_col = df[DATETIME_COL]
    if dt_col.dt.tz is not None:
        t_end = t_end.tz_localize("UTC") if t_end.tzinfo is None else t_end
        v_end = v_end.tz_localize("UTC") if v_end.tzinfo is None else v_end

    train_mask = dt_col <= t_end
    val_mask = (dt_col > t_end) & (dt_col <= v_end)
    test_mask = dt_col > v_end

    train_df = df.loc[train_mask].reset_index(drop=True)
    val_df = df.loc[val_mask].reset_index(drop=True)
    test_df = df.loc[test_mask].reset_index(drop=True)

    logger.info(
        "Chronological split complete — train: %d, val: %d, test: %d rows.",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    if len(train_df) == 0:
        raise ValueError("Training split is empty. Check train_end date.")
    if len(val_df) == 0:
        logger.warning("Validation split is empty. Check val_end date.")
    if len(test_df) == 0:
        logger.warning("Test split is empty. All data falls within val_end.")

    return train_df, val_df, test_df
