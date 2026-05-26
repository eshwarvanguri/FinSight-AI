"""Train the credit risk model.

Usage:
    python -m src.traditional_ml.train \\
        --config configs/model_config.yaml \\
        --data data/raw/lending_club.csv \\
        --output artifacts/models/credit_risk_v1.pkl \\
        --use-mlflow
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.traditional_ml.data_pipeline import (
    clean_data,
    load_data,
    save_splits,
    split_data,
)
from src.traditional_ml.feature_engineering import (
    CreditFeatureEngineer,
    FeatureSpec,
    build_preprocessing_pipeline,
    encode_target,
)
from src.utils.config import load_config, settings
from src.utils.logger import log


def build_model(model_type: str, cfg: dict[str, Any]) -> Any:
    """Build a model from config."""
    if model_type == "xgboost":
        return XGBClassifier(
            **{k: v for k, v in cfg["xgboost"].items() if k != "early_stopping_rounds"},
            random_state=cfg.get("random_state", 42),
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            **cfg["random_forest"],
            random_state=cfg.get("random_state", 42),
        )
    if model_type == "logistic":
        return LogisticRegression(**cfg["logistic"])
    if model_type == "ensemble":
        estimators = [
            ("xgb", XGBClassifier(**{k: v for k, v in cfg["xgboost"].items() if k != "early_stopping_rounds"})),
            ("rf", RandomForestClassifier(**cfg["random_forest"])),
            ("lr", LogisticRegression(**cfg["logistic"])),
        ]
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=5,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def maybe_resample(X: np.ndarray, y: np.ndarray, strategy: str, ratio: float):
    """Apply imbalance handling."""
    if strategy == "smote":
        sm = SMOTE(sampling_strategy=ratio, random_state=42)
        return sm.fit_resample(X, y)
    return X, y


def evaluate_quick(model, X, y) -> dict[str, float]:
    """Quick eval — full eval lives in evaluate.py."""
    proba = model.predict_proba(X)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
    }


def train(args: argparse.Namespace) -> None:
    """Main training entrypoint."""
    config = load_config(args.config)

    # MLflow (optional)
    mlflow = None
    if args.use_mlflow:
        import mlflow as _mlflow  # local import

        mlflow = _mlflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(args.experiment_name or config["mlflow"]["experiment_name"])
        mlflow.start_run(run_name=args.run_name)
        mlflow.log_params(config["model"][args.model_type or config["model"]["type"]])

    # 1) Load + clean
    df = load_data(args.data, nrows=args.sample)
    df = clean_data(df, target_col=config["data"]["target_column"])

    # 2) Feature engineering
    spec = FeatureSpec(
        numeric=config["data"]["numeric_features"],
        categorical=config["data"]["categorical_features"],
        derived=config["data"]["derived_features"],
        target=config["data"]["target_column"],
    )
    fe = CreditFeatureEngineer(spec)
    df = fe.fit_transform(df)

    # 3) Split
    splits = split_data(
        df,
        target_col=spec.target,
        test_size=config["data"]["test_size"],
        val_size=config["data"]["val_size"],
        stratify=config["data"]["stratify"],
    )
    save_splits(splits, settings.DATA_DIR / "processed")

    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

    feature_cols = spec.numeric + spec.categorical + [
        c for c in ["income_to_loan_ratio", "debt_to_income_v2",
                    "utilization_bucket", "credit_history_length",
                    "installment_burden"]
        if c in train_df.columns
    ]
    numeric_cols = [c for c in feature_cols if c not in spec.categorical and c != "utilization_bucket"]
    cat_cols = [c for c in spec.categorical if c in train_df.columns]
    if "utilization_bucket" in train_df.columns:
        cat_cols = list(set(cat_cols + ["utilization_bucket"]))

    X_train_raw = train_df[numeric_cols + cat_cols]
    X_val_raw = val_df[numeric_cols + cat_cols]
    y_train = encode_target(train_df[spec.target], config["data"]["positive_class"])
    y_val = encode_target(val_df[spec.target], config["data"]["positive_class"])

    log.info(f"Class balance (train): {np.bincount(y_train)}")

    # 4) Preprocessing
    preprocessor = build_preprocessing_pipeline(
        numeric_cols=numeric_cols,
        categorical_cols=cat_cols,
        scaling=config["preprocessing"]["scaling"],
    )
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)

    # 5) Imbalance
    X_train_bal, y_train_bal = maybe_resample(
        X_train, y_train,
        strategy=config["imbalance"]["strategy"],
        ratio=config["imbalance"]["sampling_strategy"],
    )
    log.info(f"After resampling: {np.bincount(y_train_bal)}")

    # 6) Build + train
    model_type = args.model_type or config["model"]["type"]
    log.info(f"Training {model_type}...")
    base_model = build_model(model_type, config["model"])
    base_model.fit(X_train_bal, y_train_bal)

    # 7) Calibration
    if config["calibration"]["method"] != "none":
        log.info(f"Calibrating with {config['calibration']['method']}")
        model = CalibratedClassifierCV(
            base_model,
            method=config["calibration"]["method"],
            cv=5,
        )
        model.fit(X_train_bal, y_train_bal)
    else:
        model = base_model

    # 8) Validation
    val_metrics = evaluate_quick(model, X_val, y_val)
    log.info(f"Validation metrics: {val_metrics}")

    if mlflow:
        mlflow.log_metrics(val_metrics)

    # 9) Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "preprocessor": preprocessor,
        "model": model,
        "feature_engineer": fe,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": cat_cols,
        "config": config,
        "val_metrics": val_metrics,
    }
    with open(output_path, "wb") as f:
        pickle.dump(artifact, f)
    log.info(f"Saved model bundle → {output_path}")

    # Save metrics JSON
    metrics_path = output_path.with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(val_metrics, f, indent=2)

    if mlflow:
        mlflow.log_artifact(str(output_path))
        mlflow.end_run()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train credit risk model")
    p.add_argument("--config", type=str, default="configs/model_config.yaml")
    p.add_argument("--data", type=str, required=True, help="Path to training CSV")
    p.add_argument("--output", type=str, default="artifacts/models/model.pkl")
    p.add_argument("--model-type", type=str, default=None,
                   choices=[None, "xgboost", "random_forest", "logistic", "ensemble"])
    p.add_argument("--use-mlflow", action="store_true")
    p.add_argument("--experiment-name", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--sample", type=int, default=None, help="Sample N rows for testing")
    p.add_argument("--tune", action="store_true", help="Run Optuna hyperparam search")
    p.add_argument("--n-trials", type=int, default=50)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
