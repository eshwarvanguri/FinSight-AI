"""Evaluate a trained credit risk model.

Usage:
    python -m src.traditional_ml.evaluate \\
        --model artifacts/models/credit_risk_v1.pkl \\
        --test-data data/processed/test.parquet \\
        --output artifacts/reports/eval_report.json \\
        --generate-shap
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.traditional_ml.feature_engineering import encode_target
from src.utils.logger import log


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """KS statistic — common in credit risk."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def find_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """F1-optimal threshold via PR curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    idx = int(np.argmax(f1[:-1])) if len(f1) > 1 else 0
    return {
        "threshold": float(thresholds[idx]) if idx < len(thresholds) else 0.5,
        "f1": float(f1[idx]),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
    }


def evaluate(args: argparse.Namespace) -> dict:
    log.info(f"Loading model from {args.model}")
    with open(args.model, "rb") as f:
        bundle = pickle.load(f)

    preprocessor = bundle["preprocessor"]
    model = bundle["model"]
    fe = bundle["feature_engineer"]
    cfg = bundle["config"]
    numeric_cols = bundle["numeric_cols"]
    cat_cols = bundle["categorical_cols"]

    # Load test data
    test_path = Path(args.test_data)
    if test_path.suffix == ".csv":
        df = pd.read_csv(test_path, low_memory=False)
    else:
        df = pd.read_parquet(test_path)

    df = fe.transform(df)
    X = df[numeric_cols + cat_cols]
    y = encode_target(df[cfg["data"]["target_column"]], cfg["data"]["positive_class"])

    X_proc = preprocessor.transform(X)
    proba = model.predict_proba(X_proc)[:, 1]
    pred = (proba >= args.threshold).astype(int)

    # Metrics
    metrics = {
        "n_samples": int(len(y)),
        "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "ks_statistic": ks_statistic(y, proba),
        "brier_score": float(brier_score_loss(y, proba)),
        "f1": float(f1_score(y, pred)),
        "threshold_used": args.threshold,
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "classification_report": classification_report(y, pred, output_dict=True),
        "optimal_threshold": find_optimal_threshold(y, proba),
    }

    log.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    log.info(f"PR-AUC:  {metrics['pr_auc']:.4f}")
    log.info(f"KS:      {metrics['ks_statistic']:.4f}")

    # SHAP
    if args.generate_shap:
        log.info("Computing SHAP values...")
        try:
            import shap

            sample_size = min(args.shap_samples, len(X_proc))
            sample_idx = np.random.RandomState(42).choice(len(X_proc), sample_size, replace=False)
            X_sample = X_proc[sample_idx]

            # For calibrated wrapper, unwrap if possible
            inner = getattr(model, "base_estimator", None) or getattr(
                model, "estimator", None
            ) or model
            explainer = shap.Explainer(inner.predict, X_sample[:100])
            shap_values = explainer(X_sample[:100])

            shap_dir = Path(args.output).parent / "shap"
            shap_dir.mkdir(parents=True, exist_ok=True)
            np.save(shap_dir / "shap_values.npy", shap_values.values)
            log.info(f"Saved SHAP values → {shap_dir}")

            mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
            metrics["top_shap_features"] = (
                np.argsort(mean_abs_shap)[-10:][::-1].tolist()
            )
        except Exception as e:
            log.warning(f"SHAP failed: {e}")

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    log.info(f"Saved evaluation → {out}")

    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate credit risk model")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--test-data", type=str, required=True)
    p.add_argument("--output", type=str, default="artifacts/reports/eval.json")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--generate-shap", action="store_true")
    p.add_argument("--shap-samples", type=int, default=1000)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
