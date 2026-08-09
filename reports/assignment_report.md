# Cats vs Dogs MLOps — Assignment 2 Report

**Course:** MLOps (S1-25_AIMLCZG523)  
**Assignment 2 — End-to-End MLOps Pipeline**  
**Topic:** Binary image classification (Cats vs Dogs) for a pet adoption platform  
**Dataset:** [Kaggle · Cats and Dogs](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)

---

## 1. Problem & Solution Overview

The objective is an *end-to-end MLOps pipeline* covering model building, experiment tracking, artifact versioning, packaging, containerization, and CI/CD-based deployment — using open-source tools.

| Component | Technology |
|---|---|
| Source & data versioning | Git, DVC |
| Experiment tracking | MLflow (filesystem store) |
| Model | PyTorch CNN baseline, PyTorch 3.x |
| Inference service | FastAPI (≥2 endpoints) |
| Containerization | Docker (multi-stage) |
| CI / CD | GitHub Actions (`.github/workflows/ci.yml`, `.github/workflows/cd.yml`) |
| Deployment | Docker Compose + Kubernetes manifests |
| Post-deploy smoke tests | `deploy/smoke_test.py` |
| Monitoring | In-app counters + JSON logs + `monitoring/simulate_traffic.py` |

---

## 2. M1 — Model Development & Experiment Traffic
### 2.1 Data & Code Versioning (Git + DVC)
- Source code is versioned in `.git`.
- The preprocessed dataset and pipeline stages are versioned with **DVC**:
  - `dvc.yaml` defines stages `preprocess -> train -> evaluate`.
  - `dvc.lock` pins hashes; `dvc repro` re-runs only out-of-date stages.
- Raw data lives under `data/raw`, processed tensors under `data/processed` (gitignored, DVC-tracked).

### 2.2 Model Building
- Baseline CNN in [`src/models/cnn_model.py`](../src/models/cnn_model.py): 5 conv-bn-pool blocks + dropout + linear head.
- Trained model saved as `models/model.pt` (PyTorch serialized).

### 2.3 Experiment Tracking (MLflow)
- `src/train.py` logs hyperparameters, loss/accuracy curves, confusion matrix, and the model artifact itself.
- Runs stored in `mlruns/` (filesystem backend; `MLFLOW_ALLOW_FILE_STORE=1`).

> See `notebooks/cats_vs_dogs_mlops_pipeline.ipynb` for a runnable cell-by-cell reproduction.

---

## 3. M2 — Model Packaging & Containerization

### 3.1 Inference Service
`api/main.py` exposes:
- `GET /health` — service + model loaded status.
- `POST /predict` — accepts an image upload, returns `{prediction, confidence, probabilities}`.
- (Bonus) `POST /predict_batch` — multi-image inference.
- (Bonus) `GET /metrics`, `GET /monitor/performance` — monitoring view.

### 3.2 Dependencies
`requirements.txt` pins versions for reproducibility (torch, torchvision, fastapi, uvicorn, mlflow, dvc, numpy, PIL, etc.).

### 3.3 Containerization
`Dockerfile`:
```dockerfile
FROM python:3.14-slim
# build-essential, libgl1, libglib2.0-0
WORKDIR /app
COPY requirements.txt .
RUN pip install ...
COPY src api monitoring models ...
HEALTHCHECK --interval=30s ... CMD uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Build & run:
```bash
docker build -t cats-vs-dogs-classifier:latest .
docker run -d --name cats-vs-dogs-api -p 8000:8000 \
  -e MODEL_PATH=models/model.pt -e DEVICE=cpu cats-vs-dogs-classifier:latest
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/predict -F "file=@data/processed/organized/cat/9733.jpg"
```

---

## 4. M3 — CI Pipeline (Test, Build, Publish)

`.github/workflows/ci.yml`: on every push/PR →
1. Check out repo.
2. Install dependencies.
3. Run unit tests (`pytest`; expectation `30 passed`, 0 warnings).
4. Build Docker image.

`.github/workflows/cd.yml`: on main →
1. Push image to **GitHub Container Registry** (`ghcr.io`).
2. Trigger deploy (Compose/K8s).

Unit tests (`tests/`): data preprocessing features + inference utilities + API + performance tracker.

---

## 5. M4 — CD Pipeline & Deployment

- **Deployment target:** Docker Compose (`deploy/docker-compose.yml`) and/or **Kubernetes** (`deploy/kubernetes/deployment.yaml`, `service.yaml`).
- Docker Compose mounts `models/` as a volume, sets `MODEL_PATH`/`DEVICE`, exposes port 8000, and adds a health check.
- Kubernetes provides declarative Deployment + Service manifests.
- **Post-deploy smoke test** `deploy/smoke_test.py`: calls `/health` + one `/predict`; fails (non-zero) if either fails → makes the release pipeline fail.

```bash
docker compose -f deploy/docker-compose.yml up -d --build
venv/bin/python deploy/smoke_test.py http://localhost:8000
```

---

## 6. M5 — Monitoring, Logs & Final Submission

### 6.1 Monitoring & Logging
- **In-app counters** (assignment permitted: logs, Prometheus, or simple in-app counters):
  - `monitoring/metrics.py` → `/metrics`: request count, latency stats, error count.
  - `api/api.log`: structured request/response logging (no raw image data).
- Docker logging via `json-file` driver.

### 6.2 Post-Deployment Performance Tracking
- `monitoring/simulate_traffic.py` sends a batch of **real images with true labels** through `/predict`, measures latency, computes live per-class accuracy, and writes `monitoring/data/live_eval.json`.
- `monitoring/performance_tracker.py` keeps a running accuracy + drift status, exposed at `/monitor/performance`.

> **Model quality (measured):** test accuracy **95.2%**, precision 95.0%, recall 95.7%, F1 95.4% (`metrics/test_metrics.json`).

---

## 7. Final Submission Artifacts

| Artifact | Path |
|---|---|
| Source code | `src/`, `tests/`, `api/`, `monitoring/`, `deploy/`, `scripts/`, `experiments/` |
| Configs | `dvc.yaml`, `dvc.lock`, `.dvc/`, `params.yaml`, `requirements.txt`, `pytest.ini` |
| Containerization | `Dockerfile`, `.dockerignore` |
| Deployment manifests | `deploy/docker-compose.yml`, `deploy/kubernetes/*.yaml` |
| Model artifact | `models/model.pt` |
| Experiments | `mlruns/` |
| Metrics | `metrics/*.json`, `monitoring/data/live_eval.json` |
| Notebook | `notebooks/cats_vs_dogs_mlops_pipeline.ipynb` |
| This documentation | `reports/` |

---

## 8. Run the Full Pipeline (from a fresh clone)

```bash
# 1. Environment
python -m venv venv && venv/bin/pip install -r requirements.txt

# 2. Data + pipeline
venv/bin/dvc repro                 # preprocess -> train -> evaluate

# 3. Tests
venv/bin/python -m pytest tests/ -q

# 4. Package + serve
docker build -t cats-vs-dogs-classifier:latest .
docker run -d --name cats-vs-dogs-api -p 8000:8000 \
  -e MODEL_PATH=models/model.pt -e DEVICE=cpu cats-vs-dogs-classifier:latest

# 5. Verify
venv/bin/python deploy/smoke_test.py http://localhost:8000

# 6. Monitor
venv/bin/python monitoring/simulate_traffic.py --n 10
cat monitoring/data/live_eval.json
```