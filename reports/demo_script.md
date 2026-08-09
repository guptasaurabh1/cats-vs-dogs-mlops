# Cats vs Dogs — Demo Script & Narration

Screen recording < 5 minutes. Sections below match M1–M5.

---

## Section 1 — M1: Data & Code Versioning (~60s)

**On screen:** terminal at `cats-vs-dogs-mlops/`

**Say:**
> "Module 1 — versioning. The dataset and code are tracked with DVC and Git. `dvc repro` runs the full pipeline: pre-processing the images into 224×224 tensors, training the CNN, and evaluating it. The pipeline produced the model artifact `models/model.pt`, and DVC records every stage in `dvc.lock`."

**Run:**
```bash
venv/bin/dvc repro
git add -A
git commit -m "M1: dataset, code, model artifact versioned"
git log --oneline
```

---

## Section 2 — M2a: Experiment Tracking with MLflow (30s)

**On screen:** browser at http://127.0.0.1:5000

**Say:**
> "Module 2 — experiment tracking. I launch MLflow and open the UI. Every training run logs hyperparameters like batch size and learning rate, metrics like validation accuracy and loss, and artifacts such as the confusion matrix and loss curves."

**Run — Terminal 1 (MLflow server):**
```bash
export MLFLOW_ALLOW_FILE_STORE=1
venv/bin/mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri ./mlruns
```
**Run — Terminal 2 (one run):**
```bash
venv/bin/python src/train.py --epochs 20 --batch-size 32 --lr 0.001 --experiment-name cats-vs-dogs-assignment
```

---

## Section 3 — M2b: Packaging & Containerization (30s)

**On screen:** Terminal

**Say:**
> "I package the model into a container. Building the Docker image with a multi-stage Dockerfile keeps it slim and reproducible. Running the container exposes the FastAPI service on port 8000. I verify the health and prediction endpoints with curl."

**Run:**
```bash
docker build -t cats-vs-dogs-classifier:latest .
docker run -d --name cats-vs-dogs-api -p 8000:8000 \
  -e MODEL_PATH=models/model.pt -e DEVICE=cpu cats-vs-dogs-classifier:latest
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/predict -F "file=@data/processed/organized/cat/9733.jpg"
```
**Expected:** `/health` → `{"status":"healthy"}` ; `/predict` → `{"prediction":"cat"}`.

---

## Section 4 — M3: CI (15s)

**Say:**
> "Module 3 — CI. Unit tests cover the data preprocessing and inference functions. `pytest` gives 30 passing tests, and the GitHub Actions workflow `.github/workflows/ci.yml` runs tests and builds the image on every push."

**Run:**
```bash
venv/bin/python -m pytest tests/ -q     # expect "30 passed"
```

---

## Section 5 — M4: Deployment (30s)

**Say:**
> "Module 4 — deployment. I deploy with Docker Compose, which mounts the model and sets a health check. Kubernetes manifests are in `deploy/kubernetes/`. Post-deploy, a smoke test calls health + prediction and fails the pipeline if anything is wrong."

**Run:**
```bash
docker compose -f deploy/docker-compose.yml up -d --build
venv/bin/python deploy/smoke_test.py http://localhost:8000   # exit 0
```

---

## Section 6 — M5: Monitoring (30s)

**Say:**
> "Module 5 — monitoring. The service logs every request and I track request count and latency. I feed real images with true labels through the API and compute live accuracy. This shows the model at ~95% — matching its validation performance."

**Run:**
```bash
venv/bin/python monitoring/simulate_traffic.py --n 10
cat monitoring/data/live_eval.json
curl -s http://localhost:8000/monitor/performance
```
**Expected:** `live_accuracy` ≈ 0.9.

---

## Section 7 — Wrap-up (10s)

**Say:**
> "Finally, package everything for submission — source, configs, Docker/Compose/K8s, CI/CD, the model, metrics, and this demo."

**Run:**
```bash
zip -r ../MLOps2_submission.zip \
  src tests api monitoring deploy .github \
  Dockerfile docker-compose.yml requirements.txt dvc.yaml params.yaml \
  models/model.pt mlruns metrics notebooks reports
```