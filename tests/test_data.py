"""
Tests for data loading and validation (src/data/).

Run with: pytest tests/test_data.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.data.validator import validate_schema
from src.data.splitter import chronological_split


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_valid_df() -> pd.DataFrame:
    """Minimal DataFrame with all required columns."""
    return pd.DataFrame({
        "game_id": ["G1", "G1", "G1"],
        "draw_id": ["D1", "D2", "D3"],
        "machine_id": ["M1", "M1", "M2"],
        "ball_set_id": ["B1", "B1", "B2"],
        "venue_id": ["V1", "V1", "V1"],
        "draw_local_datetime": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        "temperature_c": [20.5, 21.0, 19.8],
    })


@pytest.fixture
def incomplete_df() -> pd.DataFrame:
    """DataFrame missing required columns."""
    return pd.DataFrame({
        "game_id": ["G1"],
        "draw_id": ["D1"],
        # machine_id, ball_set_id, venue_id, draw_local_datetime, temperature_c missing
    })


# ---------------------------------------------------------------------------
# validate_schema tests
# ---------------------------------------------------------------------------

class TestValidateSchema:
    def test_valid_df_passes(self, minimal_valid_df):
        result = validate_schema(minimal_valid_df)
        assert result["is_valid"] is True
        assert result["missing_columns"] == []

    def test_missing_columns_detected(self, incomplete_df):
        result = validate_schema(incomplete_df)
        assert result["is_valid"] is False
        assert "machine_id" in result["missing_columns"]
        assert "draw_local_datetime" in result["missing_columns"]

    def test_empty_df_fails(self):
        result = validate_schema(pd.DataFrame())
        assert result["is_valid"] is False
        assert len(result["missing_columns"]) > 0

    def test_result_has_required_keys(self, minimal_valid_df):
        result = validate_schema(minimal_valid_df)
        assert "is_valid" in result
        assert "missing_columns" in result
        assert "warnings" in result


# ---------------------------------------------------------------------------
# chronological_split tests
# ---------------------------------------------------------------------------

class TestChronologicalSplit:
    def test_split_returns_three_sets(self, minimal_valid_df):
        train, val, test = chronological_split(
            minimal_valid_df,
            datetime_col="draw_local_datetime",
            train_end="2023-01-01",
            val_end="2023-01-02",
        )
        assert len(train) >= 0
        assert len(val) >= 0
        assert len(test) >= 0

    def test_no_overlap_between_splits(self, minimal_valid_df):
        train, val, test = chronological_split(
            minimal_valid_df,
            datetime_col="draw_local_datetime",
            train_end="2023-01-01",
            val_end="2023-01-02",
        )
        if len(train) > 0 and len(val) > 0:
            assert train["draw_local_datetime"].max() <= val["draw_local_datetime"].min()
        if len(val) > 0 and len(test) > 0:
            assert val["draw_local_datetime"].max() <= test["draw_local_datetime"].min()

    def test_total_rows_preserved(self, minimal_valid_df):
        train, val, test = chronological_split(
            minimal_valid_df,
            datetime_col="draw_local_datetime",
            train_end="2023-01-01",
            val_end="2023-01-02",
        )
        assert len(train) + len(val) + len(test) == len(minimal_valid_df)
