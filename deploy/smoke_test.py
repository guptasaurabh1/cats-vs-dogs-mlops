"""
Post-deployment smoke test script.
Calls health endpoint and one prediction endpoint to verify deployment.
Fails with non-zero exit code if checks fail.
"""

import sys
import requests
import json

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"


def test_health():
    """Test health check endpoint."""
    print(f"\n[SMOKE TEST] Health check -> {BASE_URL}/health")
    resp = requests.get(f"{BASE_URL}/health", timeout=10)
    assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
    data = resp.json()
    assert data["status"] == "healthy", f"Unhealthy: {data}"
    assert data["model_loaded"] is True, "Model not loaded"
    print(f"  PASS: {json.dumps(data, indent=2)}")
    return True


def test_prediction():
    """Test prediction endpoint with a synthetic image."""
    print(f"\n[SMOKE TEST] Prediction -> {BASE_URL}/predict")

    # Create a small test image (RGB gradient)
    import io
    from PIL import Image

    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    resp = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("test.jpg", buf, "image/jpeg")},
        timeout=30,
    )
    assert resp.status_code == 200, f"Prediction failed: {resp.status_code}"
    data = resp.json()
    assert "prediction" in data, f"Missing prediction key: {data}"
    assert data["prediction"] in ("cat", "dog"), f"Invalid prediction: {data['prediction']}"
    assert 0 <= data["confidence"] <= 1, f"Invalid confidence: {data['confidence']}"
    print(f"  PASS: {json.dumps(data, indent=2)}")
    return True


def test_metrics():
    """Test metrics endpoint."""
    print(f"\n[SMOKE TEST] Metrics -> {BASE_URL}/metrics")
    resp = requests.get(f"{BASE_URL}/metrics", timeout=10)
    assert resp.status_code == 200, f"Metrics check failed: {resp.status_code}"
    data = resp.json()
    print(f"  PASS: {json.dumps(data, indent=2)}")
    return True


def main():
    print("=" * 50)
    print("SMOKE TESTS - Cats vs Dogs Classifier")
    print(f"Target: {BASE_URL}")
    print("=" * 50)

    tests = [test_health, test_prediction, test_metrics]
    failures = 0

    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failures += 1
        except requests.exceptions.ConnectionError:
            print(f"  FAIL: Cannot connect to {BASE_URL}")
            failures += 1
        except Exception as e:
            print(f"  FAIL: Unexpected error: {e}")
            failures += 1

    print(f"\n{'=' * 50}")
    if failures == 0:
        print("ALL SMOKE TESTS PASSED")
        sys.exit(0)
    else:
        print(f"{failures} SMOKE TEST(S) FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
