"""
src/data/validator.py
---------------------
Schema validation utilities for the Machine Failure Risk project.
"""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

MANDATORY_COLUMNS: list[str] = [
    "game_id",
    "draw_id",
    "machine_id",
    "ball_set_id",
    "venue_id",
    "draw_local_datetime",
    "temperature_c",
]

NUMERIC_COLUMNS: list[str] = ["temperature_c"]


def validate_schema(df: pd.DataFrame) -> dict[str, Any]:
    """
    Validate that a DataFrame conforms to the expected schema.

    Checks for mandatory columns, looks for obvious data quality issues
    such as high null ratios, and returns a structured validation report.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.

    Returns
    -------
    dict
        A validation report with the following keys:

        - ``is_valid`` (bool): True if all mandatory columns are present.
        - ``missing_columns`` (list[str]): Names of missing mandatory columns.
        - ``warnings`` (list[str]): Non-fatal data quality warnings.
    """
    report: dict[str, Any] = {
        "is_valid": True,
        "missing_columns": [],
        "warnings": [],
    }

    # Check mandatory columns
    missing = [col for col in MANDATORY_COLUMNS if col not in df.columns]
    if missing:
        report["is_valid"] = False
        report["missing_columns"] = missing
        logger.error("Schema validation failed. Missing columns: %s", missing)

    # High-null-ratio warnings
    null_ratios = df.isnull().mean()
    for col, ratio in null_ratios.items():
        if ratio > 0.20:
            warn_msg = (
                f"Column '{col}' has {ratio:.1%} null values — consider imputation."
            )
            report["warnings"].append(warn_msg)
            logger.warning(warn_msg)

    # Numeric range warnings
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            if df[col].min() < -50 or df[col].max() > 60:
                warn_msg = (
                    f"Column '{col}' contains values outside plausible "
                    f"temperature range [-50, 60]°C."
                )
                report["warnings"].append(warn_msg)

    # Duplicate draw check
    if "draw_id" in df.columns:
        dup_count = df["draw_id"].duplicated().sum()
        if dup_count > 0:
            warn_msg = f"Found {dup_count} duplicate draw_id values."
            report["warnings"].append(warn_msg)
            logger.warning(warn_msg)

    logger.info(
        "Schema validation complete. is_valid=%s, warnings=%d",
        report["is_valid"],
        len(report["warnings"]),
    )
    return report
