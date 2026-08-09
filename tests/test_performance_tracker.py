"""
Unit tests for the post-deployment performance tracker.
Covers record_prediction, annotate, get_accuracy, get_drift_status.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitoring.performance_tracker import PerformanceTracker


@pytest.fixture
def tracker(tmp_path):
    """Fresh tracker writing to a temp file."""
    return PerformanceTracker(data_path=str(tmp_path / "perf.json"))


class TestPerformanceTracker:
    def test_record_prediction_with_true_label(self, tracker):
        tracker.record_prediction(
            input_id="img_1",
            predicted_label="cat",
            confidence=0.94,
            true_label="cat",
        )
        acc = tracker.get_accuracy()
        assert acc["total_samples"] == 1
        assert acc["accuracy"] == 1.0

    def test_unlabeled_predictions_excluded_from_accuracy(self, tracker):
        tracker.record_prediction(
            input_id="img_1",
            predicted_label="cat",
            confidence=0.94,
            true_label=None,
        )
        acc = tracker.get_accuracy()
        # No labeled records yet -> returns the placeholder summary
        assert acc.get("labeled_samples", 0) == 1
        assert acc.get("total_samples", 0) == 0

    def test_annotate_adds_ground_truth(self, tracker):
        tracker.record_prediction(
            input_id="img_5",
            predicted_label="dog",
            confidence=0.80,
            true_label=None,
        )
        assert tracker.annotate("img_5", "dog") is True
        acc = tracker.get_accuracy()
        assert acc["total_samples"] == 1
        assert acc["accuracy"] == 1.0

    def test_annotate_missing_id(self, tracker):
        tracker.record_prediction(
            input_id="img_5",
            predicted_label="dog",
            confidence=0.80,
            true_label=None,
        )
        assert tracker.annotate("nope", "dog") is False

    def test_drift_insufficient_data(self, tracker):
        for i in range(10):
            tracker.record_prediction(
                input_id=f"img_{i}",
                predicted_label="cat",
                confidence=0.9,
                true_label="cat",
            )
        drift = tracker.get_drift_status()
        assert drift["status"] == "insufficient_data"
        assert drift["drift_detected"] is False

    def test_drift_healthy(self, tracker):
        for i in range(50):
            tracker.record_prediction(
                input_id=f"img_{i}",
                predicted_label="cat",
                confidence=0.9,
                true_label="cat",
            )
        drift = tracker.get_drift_status(baseline_accuracy=0.95)
        assert drift["status"] == "healthy"
        assert drift["drift_detected"] is False

    def test_drift_detected(self, tracker):
        # All wrong -> accuracy 0.0, big drop from 0.95 baseline
        for i in range(50):
            tracker.record_prediction(
                input_id=f"img_{i}",
                predicted_label="cat",
                confidence=0.6,
                true_label="dog",
            )
        drift = tracker.get_drift_status(baseline_accuracy=0.95)
        assert drift["status"] == "drift_detected"
        assert drift["drift_detected"] is True


class TestSimulateRequests:
    def test_simulate_requests_baseline(self, tmp_path):
        tracker = PerformanceTracker(data_path=str(tmp_path / "sim.json"))
        metrics = tracker.simulate_requests(100, accuracy_drift=0.0)
        assert metrics["total_samples"] == 100
        # Base accuracy 0.92, allow slack
        assert 0.80 <= metrics["accuracy"] <= 1.0

    def test_simulate_requests_drift(self, tmp_path):
        tracker = PerformanceTracker(data_path=str(tmp_path / "sim2.json"))
        metrics = tracker.simulate_requests(100, accuracy_drift=0.5)
        # Heavy drift -> accuracy should be around 0.42, well below baseline
        assert metrics["accuracy"] < 0.80
