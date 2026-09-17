"""
Tests for anomaly detection models (src/models/).

Run with: pytest tests/test_models.py -v
"""

import pytest
import numpy as np
import os
import tempfile

from src.models.baseline_zscore import RobustZScoreDetector
from src.models.isolation_forest import IsolationForestDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_normal_data() -> np.ndarray:
    """Small, clean training dataset (100 samples, 5 features)."""
    rng = np.random.default_rng(42)
    return rng.normal(loc=0.0, scale=1.0, size=(100, 5))


@pytest.fixture
def dummy_test_data() -> np.ndarray:
    """Test dataset with some injected anomalies."""
    rng = np.random.default_rng(99)
    normal = rng.normal(loc=0.0, scale=1.0, size=(18, 5))
    anomalies = rng.normal(loc=10.0, scale=1.0, size=(2, 5))  # clear anomalies
    return np.vstack([normal, anomalies])


# ---------------------------------------------------------------------------
# RobustZScoreDetector tests
# ---------------------------------------------------------------------------

class TestRobustZScoreDetector:
    def test_fit_does_not_raise(self, dummy_normal_data):
        detector = RobustZScoreDetector()
        detector.fit(dummy_normal_data)  # Should not raise

    def test_score_returns_correct_shape(self, dummy_normal_data, dummy_test_data):
        detector = RobustZScoreDetector()
        detector.fit(dummy_normal_data)
        scores = detector.score(dummy_test_data)
        assert scores.shape == (len(dummy_test_data),)

    def test_predict_returns_binary_labels(self, dummy_normal_data, dummy_test_data):
        detector = RobustZScoreDetector()
        detector.fit(dummy_normal_data)
        preds = detector.predict(dummy_test_data, threshold=3.0)
        assert set(np.unique(preds)).issubset({-1, 1})

    def test_anomalies_score_higher(self, dummy_normal_data, dummy_test_data):
        detector = RobustZScoreDetector()
        detector.fit(dummy_normal_data)
        scores = detector.score(dummy_test_data)
        normal_scores = scores[:18]
        anomaly_scores = scores[18:]
        assert anomaly_scores.mean() > normal_scores.mean()

    def test_score_before_fit_raises(self, dummy_test_data):
        detector = RobustZScoreDetector()
        with pytest.raises(RuntimeError, match="not been fitted"):
            detector.score(dummy_test_data)


# ---------------------------------------------------------------------------
# IsolationForestDetector tests
# ---------------------------------------------------------------------------

class TestIsolationForestDetector:
    def test_fit_does_not_raise(self, dummy_normal_data):
        detector = IsolationForestDetector(n_estimators=10, random_state=42)
        detector.fit(dummy_normal_data)

    def test_score_returns_correct_shape(self, dummy_normal_data, dummy_test_data):
        detector = IsolationForestDetector(n_estimators=10, random_state=42)
        detector.fit(dummy_normal_data)
        scores = detector.score(dummy_test_data)
        assert scores.shape == (len(dummy_test_data),)

    def test_predict_returns_binary_labels(self, dummy_normal_data, dummy_test_data):
        detector = IsolationForestDetector(n_estimators=10, random_state=42)
        detector.fit(dummy_normal_data)
        preds = detector.predict(dummy_test_data)
        assert set(np.unique(preds)).issubset({-1, 1})

    def test_save_and_load(self, dummy_normal_data, dummy_test_data):
        detector = IsolationForestDetector(n_estimators=10, random_state=42)
        detector.fit(dummy_normal_data)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_model.joblib")
            detector.save(path)
            loaded = IsolationForestDetector.load(path)
            scores_original = detector.score(dummy_test_data)
            scores_loaded = loaded.score(dummy_test_data)
            np.testing.assert_array_almost_equal(scores_original, scores_loaded)

    def test_deterministic_with_same_seed(self, dummy_normal_data, dummy_test_data):
        d1 = IsolationForestDetector(n_estimators=20, random_state=0)
        d2 = IsolationForestDetector(n_estimators=20, random_state=0)
        d1.fit(dummy_normal_data)
        d2.fit(dummy_normal_data)
        np.testing.assert_array_equal(d1.score(dummy_test_data), d2.score(dummy_test_data))

    def test_score_before_fit_raises(self, dummy_test_data):
        detector = IsolationForestDetector()
        with pytest.raises(RuntimeError, match="not been fitted"):
            detector.score(dummy_test_data)
