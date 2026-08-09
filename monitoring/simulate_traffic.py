#!/usr/bin/env python3
"""
Post-deployment monitoring: send a batch of real images through the live
inference API, annotate them with their true labels, and evaluate live
accuracy + drift (M5 requirement: collect a batch of simulated requests
with true labels).

Usage:
    python monitoring/simulate_traffic.py [--url http://localhost:8000] [--n 10]
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import requests


CLASS_DIRS = {
    "cat": "data/processed/organized/cat",
    "dog": "data/processed/organized/dog",
}


def collect_sample_images(n_per_class: int) -> list[tuple[str, str, bytes]]:
    """Grab n random images per class from the organized dataset.

    Returns the original on-disk file bytes (like a real client would send),
    so the API's preprocessing operates on the exact same data it will see in
    production. Re-encoding PNG -> JPEG here would corrupt the input and drag
    down measured accuracy.
    """
    samples = []
    for label, img_dir in CLASS_DIRS.items():
        d = Path(img_dir)
        if not d.exists():
            print(f"  [warn] {d} not found, skipping {label}")
            continue
        files = sorted(d.iterdir())
        random.seed(42)
        chosen = random.sample(files, min(n_per_class, len(files)))
        for f in chosen:
            try:
                samples.append((label, f.name, f.read_bytes()))
            except Exception:
                continue
    return samples


def run_traffic(base_url: str, n_per_class: int):
    print(f"Collecting {n_per_class} images per class...")
    samples = collect_sample_images(n_per_class)
    if not samples:
        print("ERROR: no images found. Run src/data/download.py + preprocess.py first.")
        return 1

    print(f"{len(samples)} images collected. Sending to {base_url}/predict ...")
    correct = 0
    results = []
    for i, (true_label, filename, file_bytes) in enumerate(samples):
        t0 = time.time()
        resp = requests.post(
            f"{base_url}/predict",
            files={"file": (filename, file_bytes, "image/jpeg")},
            timeout=60,
        )
        latency = (time.time() - t0) * 1000
        if resp.status_code != 200:
            print(f"  [error] request {i} failed: {resp.status_code} {resp.text[:100]}")
            continue
        pred = resp.json()["prediction"]
        is_correct = pred == true_label
        correct += int(is_correct)
        results.append({
            "input_id": filename,
            "true_label": true_label,
            "predicted_label": pred,
            "correct": is_correct,
            "confidence": resp.json()["confidence"],
            "latency_ms": round(latency, 2),
        })

    n = len(results)
    accuracy = correct / n if n else 0.0

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_images": n,
        "n_correct": correct,
        "live_accuracy": round(accuracy, 4),
        "per_class": {},
    }
    for cls in ("cat", "dog"):
        cls_results = [r for r in results if r["true_label"] == cls]
        cls_correct = sum(1 for r in cls_results if r["correct"])
        report["per_class"][cls] = {
            "n": len(cls_results),
            "correct": cls_correct,
            "accuracy": round(cls_correct / max(len(cls_results), 1), 4),
        }
    report["avg_latency_ms"] = round(
        sum(r["latency_ms"] for r in results) / max(n, 1), 2
    )

    out = Path("monitoring/data/live_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nResult saved to {out}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Send live evaluation traffic to API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--n", type=int, default=10, help="images per class to test")
    args = parser.parse_args()
    return run_traffic(args.url, args.n)


if __name__ == "__main__":
    sys.exit(main())
