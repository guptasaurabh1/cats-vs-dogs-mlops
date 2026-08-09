"""
Post-deployment model performance tracker.
M5 requirement: collect simulated requests with true labels for accuracy monitoring.
"""

import json
import random
import time
from pathlib import Path


class PerformanceTracker:
    """
    Tracks model performance after deployment by storing
    prediction results alongside ground truth labels.
    """

    def __init__(self, data_path: str = "monitoring/data/performance.json"):
        self.data_path = Path(data_path)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self.records = []
        self._load()

    def record_prediction(
        self,
        input_id: str,
        predicted_label: str,
        confidence: float,
        true_label: str = None,
        latency_ms: float = None,
    ):
        """Record a single prediction with an optional ground truth label.

        At request time the true label is unknown (true_label=None); it can
        be attached later via {annotate} once human feedback arrives, which
        is the standard production monitoring loop.
        """
        record = {
            "timestamp": time.time(),
            "input_id": input_id,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "correct": (true_label == predicted_label) if true_label is not None else None,
            "confidence": confidence,
            "latency_ms": latency_ms,
        }
        self.records.append(record)

        # Persist every 10 records
        if len(self.records) % 10 == 0:
            self._persist()

    def annotate(self, input_id: str, true_label: str) -> bool:
        """Attach a ground-truth label to a previously recorded prediction."""
        for rec in self.records:
            if rec["input_id"] == input_id:
                rec["true_label"] = true_label
                rec["correct"] = rec["predicted_label"] == true_label
                self._persist()
                return True
        return False

    def get_accuracy(self) -> dict:
        """Compute current accuracy metrics from labeled records."""
        labeled = [r for r in self.records if r["true_label"] is not None]
        if not labeled:
            return {
                "total_samples": 0,
                "labeled_samples": len(self.records),
                "note": "no labeled records yet",
            }

        confs = [r["confidence"] for r in self.records]

        total = len(labeled)
        correct = sum(1 for r in labeled if r["correct"])

        # Per-class accuracy
        cats = [r for r in labeled if r["true_label"] == "cat"]
        dogs = [r for r in labeled if r["true_label"] == "dog"]
        cat_acc = sum(1 for r in cats if r["correct"]) / max(len(cats), 1)
        dog_acc = sum(1 for r in dogs if r["correct"]) / max(len(dogs), 1)

        return {
            "total_samples": total,
            "labeled_samples": len(self.records),
            "accuracy": round(correct / total, 4),
            "cat_accuracy": round(cat_acc, 4),
            "dog_accuracy": round(dog_acc, 4),
            "avg_confidence": round(sum(confs) / len(confs), 4),
            "confusion": {
                "cat_as_cat": sum(1 for r in cats if r["predicted_label"] == "cat"),
                "cat_as_dog": sum(1 for r in cats if r["predicted_label"] == "dog"),
                "dog_as_cat": sum(1 for r in dogs if r["predicted_label"] == "cat"),
                "dog_as_dog": sum(1 for r in dogs if r["predicted_label"] == "dog"),
            },
        }

    def get_drift_status(self, baseline_accuracy: float = 0.95, threshold: float = 0.05) -> dict:
        """Compare current live accuracy to the offline baseline.

        If live accuracy drops more than `threshold` below the baseline,
        the model is flagged for retraining (data/model drift).
        """
        accuracy = self.get_accuracy()
        if not accuracy or accuracy.get("labeled_samples", 0) < 30:
            return {
                "status": "insufficient_data",
                "baseline_accuracy": baseline_accuracy,
                "current_accuracy": None,
                "drift_detected": False,
                "note": "Need at least 30 labeled predictions to evaluate drift.",
            }

        current = accuracy["accuracy"]
        drift = baseline_accuracy - current

        return {
            "status": "drift_detected" if drift > threshold else "healthy",
            "baseline_accuracy": baseline_accuracy,
            "current_accuracy": current,
            "accuracy_drop": round(drift, 4),
            "threshold": threshold,
            "drift_detected": drift > threshold,
        }

    def simulate_requests(self, n_requests: int = 100, accuracy_drift: float = 0.0):
        """
        Generate simulated prediction records to test monitoring.
        accuracy_drift: reduce accuracy to simulate performance degradation.
        """
        base_accuracy = 0.92
        effective_accuracy = base_accuracy - accuracy_drift

        for i in range(n_requests):
            true_label = random.choice(["cat", "dog"])
            is_correct = random.random() < effective_accuracy
            predicted_label = true_label if is_correct else (
                "dog" if true_label == "cat" else "cat"
            )
            confidence = random.uniform(0.75, 0.99) if is_correct else random.uniform(0.50, 0.70)
            latency = random.uniform(15, 80)

            self.record_prediction(
                input_id=f"sim_{i:04d}",
                predicted_label=predicted_label,
                confidence=round(confidence, 4),
                true_label=true_label,
                latency_ms=round(latency, 2),
            )

        return self.get_accuracy()

    def _persist(self):
        """Save records to disk."""
        data = {
            "last_updated": time.time(),
            "total_records": len(self.records),
            "metrics": self.get_accuracy(),
            "recent_records": self.records[-100:],  # Keep last 100 in file
        }
        with open(self.data_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load existing records from disk."""
        try:
            if self.data_path.exists():
                with open(self.data_path) as f:
                    data = json.load(f)
                self.records = data.get("recent_records", [])
        except Exception:
            self.records = []


if __name__ == "__main__":
    # Demo: simulate 200 requests and show performance metrics
    tracker = PerformanceTracker()
    metrics = tracker.simulate_requests(200, accuracy_drift=0.0)
    print("Performance Metrics (baseline):")
    print(json.dumps(metrics, indent=2))

    # Simulate drift
    tracker2 = PerformanceTracker()
    drift_metrics = tracker2.simulate_requests(200, accuracy_drift=0.15)
    print("\nPerformance Metrics (with drift):")
    print(json.dumps(drift_metrics, indent=2))
