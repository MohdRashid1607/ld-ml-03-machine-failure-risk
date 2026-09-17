"""
Rolling-origin backtest for chronological evaluation.

Performs a walk-forward backtest: trains on an expanding or sliding window
and scores a held-out future window — respecting temporal order strictly.

LEAKAGE WARNING: The test period must never be seen during fitting.
"""

import numpy as np
import pandas as pd
from typing import Callable, Optional


def rolling_origin_backtest(
    df: pd.DataFrame,
    feature_col: str,
    datetime_col: str,
    model_factory: Callable,
    window_size: int = 90,
    step_size: int = 30,
    score_window: int = 30,
) -> pd.DataFrame:
    """
    Run a rolling-origin (walk-forward) backtest.

    For each origin, trains the model on 'window_size' days of history and
    scores the next 'score_window' days. Returns a DataFrame of results.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset sorted by datetime_col (ascending).
    feature_col : str
        Name of the numeric feature column to use (single feature for simplicity).
        For multivariate backtesting, adapt to pass a feature matrix.
    datetime_col : str
        Name of the datetime column.
    model_factory : Callable
        A callable that returns a fresh, unfitted model instance when called
        with no arguments. E.g., lambda: IsolationForestDetector().
    window_size : int
        Number of rows (draws) to use for each training window.
    step_size : int
        Number of rows to advance the origin between iterations.
    score_window : int
        Number of rows to score after each training window.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: origin_end_date, n_train, n_scored,
        mean_score, n_anomalies, flag_rate per backtest iteration.
    """
    df = df.sort_values(datetime_col).reset_index(drop=True)
    results = []

    n = len(df)
    start = window_size

    while start + score_window <= n:
        train_idx = slice(start - window_size, start)
        score_idx = slice(start, start + score_window)

        X_train = df.iloc[train_idx][[feature_col]].values
        X_score = df.iloc[score_idx][[feature_col]].values
        origin_end_date = df.iloc[start - 1][datetime_col]

        model = model_factory()
        model.fit(X_train)
        scores = model.score(X_score)
        preds = model.predict(X_score)

        n_anomalies = int((preds == -1).sum())
        results.append({
            "origin_end_date": origin_end_date,
            "n_train": len(X_train),
            "n_scored": len(X_score),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "n_anomalies": n_anomalies,
            "flag_rate": float(n_anomalies / len(X_score)),
        })

        start += step_size

    return pd.DataFrame(results)
