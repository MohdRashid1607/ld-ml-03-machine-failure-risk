"""
src/features/pipeline.py
--------------------------
Master feature engineering pipeline.

Calls all feature modules in the correct order and returns a single,
combined feature matrix ready for model training or inference.
"""

import logging

import pandas as pd

from src.features.data_quality import compute_quality_features
from src.features.distribution_diagnostics import compute_distribution_features
from src.features.environment import compute_environment_features
from src.features.machine_history import compute_machine_history_features
from src.features.temporal_stability import compute_temporal_features

logger = logging.getLogger(__name__)


def build_feature_matrix(
    df: pd.DataFrame,
    window: int = 30,
    fit: bool = True,
) -> pd.DataFrame:
    """
    Build a complete feature matrix by running all feature engineering modules.

    Pipeline order
    --------------
    1. Machine history features (cumulative usage, time-since-service, etc.)
    2. Distribution diagnostic features (rolling mean, variance, range, etc.)
    3. Temporal stability features (rolling z-score, drift, anomaly count)
    4. Environmental / calendar features (hour, day, month, season, temp mean)
    5. Data quality meta-features (missingness ratio and flag)

    All modules enforce strict temporal safety (no look-ahead leakage) by
    using shift(1) before rolling windows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw (or minimally cleaned) input DataFrame. Must be sorted
        chronologically by ``draw_local_datetime``.
    window : int, optional
        Rolling window size passed to distribution and temporal modules.
        Defaults to 30.
    fit : bool, optional
        Reserved for future transformer-based feature steps that may need
        fit vs. inference modes (e.g., fitted encoders). Currently unused
        by the rolling-only modules. Defaults to True.

    Returns
    -------
    pd.DataFrame
        The input DataFrame enriched with all computed features. Original
        columns are preserved; new feature columns are appended.
    """
    logger.info("Building feature matrix: %d rows × %d cols (input).", *df.shape)

    df = compute_machine_history_features(df)
    logger.debug("After machine_history: %d cols.", df.shape[1])

    df = compute_distribution_features(df, window=window)
    logger.debug("After distribution_diagnostics: %d cols.", df.shape[1])

    df = compute_temporal_features(df, window=window)
    logger.debug("After temporal_stability: %d cols.", df.shape[1])

    df = compute_environment_features(df)
    logger.debug("After environment: %d cols.", df.shape[1])

    df = compute_quality_features(df)
    logger.debug("After data_quality: %d cols.", df.shape[1])

    logger.info(
        "Feature matrix built: %d rows × %d cols (output).", *df.shape
    )
    return df
