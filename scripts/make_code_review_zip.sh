#!/usr/bin/env bash
# Create a light code-review zip (~means exclude the heavy generated files).
# Generated for Claude review; the 4.3GB tensors / 19MB model / mlruns are
# left out so the zip stays a few MB and is uploadable.

set -euo pipefail
cd "$(dirname "$0")/.."   # project root (cats-vs-dogs-mlops)

OUT="${1:-MLOps_code_review.zip}"
rm -f "$OUT"

zip -r "$OUT" \
  src tests api monitoring deploy scripts experiments .github \
  notebooks reports \
  Dockerfile docker-compose.yml requirements.txt pytest.ini \
  params.yaml dvc.yaml dvc.lock \
  models/model.pt \
  -x "*/__pycache__/*" "*/node_modules/*" "*.pyc"

echo "Created $OUT"
ls -lh "$OUT"