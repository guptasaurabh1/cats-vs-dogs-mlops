"""
Request/response metrics collector for the inference service.
M5 requirement: track request count, latency, and error rate.
"""

import time
import json
import threading
from collections import defaultdict
from pathlib import Path


class MetricsCollector:
    """Thread-safe in-memory metrics collector with periodic persistence."""

    def __init__(self, persist_path: str = "monitoring/data/metrics.json"):
        self._lock = threading.RLock()
        self.persist_path = Path(persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        self.request_counts = defaultdict(int)
        self.endpoint_latencies = defaultdict(list)
        self.error_count = 0
        self.total_requests = 0
        self.start_time = time.time()

        # Load persisted state if exists
        self._load()

    def record_request(self, endpoint: str, latency: float = None):
        """Record a request to an endpoint with optional latency."""
        with self._lock:
            self.request_counts[endpoint] += 1
            self.total_requests += 1
            if latency is not None:
                self.endpoint_latencies[endpoint].append(latency)
            self._persist()

    def record_error(self):
        """Record an error."""
        with self._lock:
            self.error_count += 1
            self._persist()

    def get_summary(self) -> dict:
        """Get a summary of collected metrics."""
        with self._lock:
            uptime = time.time() - self.start_time
            summary = {
                "service": "cats-vs-dogs-classifier",
                "uptime_seconds": round(uptime, 2),
                "uptime_human": self._format_uptime(uptime),
                "total_requests": self.total_requests,
                "error_count": self.error_count,
                "error_rate": round(self.error_count / max(self.total_requests, 1), 6),
                "endpoints": {},
            }

            for endpoint, count in self.request_counts.items():
                latencies = self.endpoint_latencies.get(endpoint, [])
                endpoint_metrics = {
                    "request_count": count,
                }
                if latencies:
                    endpoint_metrics["latency"] = {
                        "mean_ms": round(sum(latencies) / len(latencies) * 1000, 2),
                        "min_ms": round(min(latencies) * 1000, 2),
                        "max_ms": round(max(latencies) * 1000, 2),
                        "p95_ms": round(self._percentile(latencies, 95) * 1000, 2),
                        "p99_ms": round(self._percentile(latencies, 99) * 1000, 2),
                    }
                summary["endpoints"][endpoint] = endpoint_metrics

            summary["requests_per_minute"] = round(
                self.total_requests / max(uptime / 60, 1), 2
            )
            return summary

    def _persist(self):
        """Save metrics to disk for durability."""
        try:
            summary = self.get_summary()
            with open(self.persist_path, "w") as f:
                json.dump(summary, f, indent=2)
        except Exception:
            pass  # Persistence failure should not crash the service

    def _load(self):
        """Load metrics from disk."""
        try:
            if self.persist_path.exists():
                with open(self.persist_path) as f:
                    json.load(f)  # Validate file is readable
        except Exception:
            pass

    @staticmethod
    def _percentile(data: list, p: float) -> float:
        """Compute the p-th percentile of a list."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100.0
        f = int(k)
        c = k - f
        if f + 1 < len(sorted_data):
            return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
        return sorted_data[-1]

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        """Format uptime as human-readable string."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours}h {minutes}m {secs}s"
