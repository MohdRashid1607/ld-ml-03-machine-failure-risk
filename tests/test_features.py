"""
Tests for feature engineering (src/features/).

Run with: pytest tests/test_features.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.machine_history import compute_machine_history_features
from src.features.environment import compute_environment_features
from src.features.data_quality import compute_quality_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Sample DataFrame resembling real lottery draw data."""
    n = 30
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n)]
    machines = ["M1"] * 15 + ["M2"] * 15
    return pd.DataFrame({
        "game_id": ["G1"] * n,
        "draw_id": [f"D{i}" for i in range(n)],
        "machine_id": machines,
        "ball_set_id": ["B1"] * n,
        "venue_id": ["V1"] * n,
        "draw_local_datetime": pd.to_datetime(dates),
        "temperature_c": np.random.default_rng(1).normal(20, 3, n),
        "num1": np.random.default_rng(2).integers(1, 50, n),
    })


# ---------------------------------------------------------------------------
# Machine history feature tests
# ---------------------------------------------------------------------------

class TestMachineHistoryFeatures:
    def test_adds_expected_columns(self, sample_df):
        result = compute_machine_history_features(sample_df)
        assert "draw_count_since_install" in result.columns

    def test_no_future_leakage(self, sample_df):
        """
        Verify that the draw_count_since_install for row i is based only on
        rows before i. That is, at row 0 for each machine, count should be 1.
        """
        result = compute_machine_history_features(sample_df)
        machine_groups = result.groupby("machine_id")
        for _, group in machine_groups:
            group = group.sort_values("draw_local_datetime").reset_index(drop=True)
            # First draw count should be 1 (only itself counted up to that point)
            assert group["draw_count_since_install"].iloc[0] == 1

    def test_count_increases_monotonically_per_machine(self, sample_df):
        result = compute_machine_history_features(sample_df)
        for machine_id, group in result.groupby("machine_id"):
            group = group.sort_values("draw_local_datetime").reset_index(drop=True)
            counts = group["draw_count_since_install"].values
            assert all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1)), \
                f"Draw count not monotone for machine {machine_id}"

    def test_output_has_same_row_count(self, sample_df):
        result = compute_machine_history_features(sample_df)
        assert len(result) == len(sample_df)


# ---------------------------------------------------------------------------
# Environment feature tests
# ---------------------------------------------------------------------------

class TestEnvironmentFeatures:
    def test_adds_temporal_columns(self, sample_df):
        result = compute_environment_features(sample_df)
        assert "hour_of_day" in result.columns
        assert "day_of_week" in result.columns
        assert "month" in result.columns
        assert "season" in result.columns

    def test_hour_in_valid_range(self, sample_df):
        result = compute_environment_features(sample_df)
        assert result["hour_of_day"].between(0, 23).all()

    def test_month_in_valid_range(self, sample_df):
        result = compute_environment_features(sample_df)
        assert result["month"].between(1, 12).all()

    def test_season_values_are_valid(self, sample_df):
        result = compute_environment_features(sample_df)
        valid_seasons = {"Spring", "Summer", "Autumn", "Winter"}
        assert set(result["season"].unique()).issubset(valid_seasons)


# ---------------------------------------------------------------------------
# Data quality feature tests
# ---------------------------------------------------------------------------

class TestDataQualityFeatures:
    def test_adds_missingness_columns(self, sample_df):
        result = compute_quality_features(sample_df)
        assert "missingness_ratio" in result.columns
        assert "missingness_flag" in result.columns

    def test_full_data_has_zero_missingness(self, sample_df):
        result = compute_quality_features(sample_df)
        assert (result["missingness_ratio"] == 0.0).all()

    def test_partial_missing_data_flagged(self, sample_df):
        df_with_nan = sample_df.copy()
        df_with_nan.loc[0, "temperature_c"] = np.nan
        result = compute_quality_features(df_with_nan)
        assert result.loc[0, "missingness_flag"] == 1
