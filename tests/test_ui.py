"""
Tests for Gradio UI helper functions (src/app.py utilities).

Run with: pytest tests/test_ui.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


# ---------------------------------------------------------------------------
# Inline UI validation helpers (mirrors logic from app.py)
# These are tested independently to allow unit testing without launching Gradio.
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "game_id",
    "draw_id",
    "machine_id",
    "ball_set_id",
    "venue_id",
    "draw_local_datetime",
    "temperature_c",
]

MAX_FILE_SIZE_ROWS = 100_000


def validate_uploaded_file(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validate an uploaded DataFrame for use in the Gradio app.

    Returns
    -------
    tuple[bool, str]
        (is_valid, message)
    """
    if df is None or len(df) == 0:
        return False, "❌ Uploaded file is empty."

    if len(df) > MAX_FILE_SIZE_ROWS:
        return False, f"❌ File too large: {len(df)} rows (max {MAX_FILE_SIZE_ROWS})."

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"❌ Missing required columns: {', '.join(missing)}"

    # Check for future timestamps (using current date as cutoff)
    if "draw_local_datetime" in df.columns:
        try:
            dt_col = pd.to_datetime(df["draw_local_datetime"], errors="coerce")
            if dt_col.isna().any():
                return False, "❌ 'draw_local_datetime' contains invalid or unparseable values."
            future_mask = dt_col > pd.Timestamp.now()
            if future_mask.any():
                return False, f"❌ File contains {future_mask.sum()} future timestamps."
        except Exception as e:
            return False, f"❌ Error parsing datetime column: {e}"

    return True, "✅ File validated successfully."


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_df() -> pd.DataFrame:
    return pd.DataFrame({
        "game_id": ["G1", "G1"],
        "draw_id": ["D1", "D2"],
        "machine_id": ["M1", "M1"],
        "ball_set_id": ["B1", "B1"],
        "venue_id": ["V1", "V1"],
        "draw_local_datetime": ["2023-01-01", "2023-06-15"],
        "temperature_c": [20.5, 22.0],
    })


@pytest.fixture
def missing_columns_df() -> pd.DataFrame:
    return pd.DataFrame({
        "game_id": ["G1"],
        "draw_id": ["D1"],
        # All other required columns missing
    })


@pytest.fixture
def future_timestamps_df() -> pd.DataFrame:
    return pd.DataFrame({
        "game_id": ["G1"],
        "draw_id": ["D1"],
        "machine_id": ["M1"],
        "ball_set_id": ["B1"],
        "venue_id": ["V1"],
        "draw_local_datetime": ["2099-01-01"],  # Future date
        "temperature_c": [20.0],
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestValidateUploadedFile:
    def test_valid_file_passes(self, valid_df):
        is_valid, msg = validate_uploaded_file(valid_df)
        assert is_valid is True
        assert "✅" in msg

    def test_empty_dataframe_rejected(self):
        is_valid, msg = validate_uploaded_file(pd.DataFrame())
        assert is_valid is False
        assert "empty" in msg.lower()

    def test_none_rejected(self):
        is_valid, msg = validate_uploaded_file(None)
        assert is_valid is False

    def test_missing_columns_rejected(self, missing_columns_df):
        is_valid, msg = validate_uploaded_file(missing_columns_df)
        assert is_valid is False
        assert "machine_id" in msg

    def test_future_timestamps_rejected(self, future_timestamps_df):
        is_valid, msg = validate_uploaded_file(future_timestamps_df)
        assert is_valid is False
        assert "future" in msg.lower()

    def test_oversized_file_rejected(self):
        big_df = pd.DataFrame({
            "game_id": ["G1"] * 200_001,
            "draw_id": [str(i) for i in range(200_001)],
            "machine_id": ["M1"] * 200_001,
            "ball_set_id": ["B1"] * 200_001,
            "venue_id": ["V1"] * 200_001,
            "draw_local_datetime": ["2023-01-01"] * 200_001,
            "temperature_c": [20.0] * 200_001,
        })
        is_valid, msg = validate_uploaded_file(big_df)
        assert is_valid is False
        assert "large" in msg.lower()
