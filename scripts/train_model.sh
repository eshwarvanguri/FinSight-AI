#!/usr/bin/env bash
# Train the credit risk model end-to-end
set -euo pipefail

echo "→ Downloading synthetic training data..."
python scripts/download_data.py --dataset lending_club --n 20000

echo "→ Training model..."
python -m src.traditional_ml.train \
    --config configs/model_config.yaml \
    --data data/raw/lending_club.csv \
    --output artifacts/models/credit_risk_v1.pkl \
    --model-type ensemble

echo "→ Evaluating on held-out test set..."
python -m src.traditional_ml.evaluate \
    --model artifacts/models/credit_risk_v1.pkl \
    --test-data data/processed/test.parquet \
    --output artifacts/reports/eval_report.json \
    --generate-shap

echo "✓ Done. Model: artifacts/models/credit_risk_v1.pkl"
