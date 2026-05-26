"""Prediction interface for the credit risk model.

Usage:
    from src.traditional_ml.predict import CreditRiskPredictor
    p = CreditRiskPredictor.load("artifacts/models/credit_risk_v1.pkl")
    p.predict({"loan_amnt": 10000, "annual_inc": 60000, ...})
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import log


class CreditRiskPredictor:
    """Inference wrapper around the saved training bundle."""

    def __init__(self, bundle: dict[str, Any]):
        self.preprocessor = bundle["preprocessor"]
        self.model = bundle["model"]
        self.feature_engineer = bundle["feature_engineer"]
        self.numeric_cols = bundle["numeric_cols"]
        self.categorical_cols = bundle["categorical_cols"]
        self.config = bundle["config"]

    @classmethod
    def load(cls, path: str | Path) -> "CreditRiskPredictor":
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        log.info(f"Loaded model bundle from {path}")
        return cls(bundle)

    def _prepare(self, X: pd.DataFrame) -> np.ndarray:
        X = self.feature_engineer.transform(X)
        # Make sure all expected columns are present
        for col in self.numeric_cols + self.categorical_cols:
            if col not in X.columns:
                X[col] = np.nan
        return self.preprocessor.transform(X[self.numeric_cols + self.categorical_cols])

    def predict_proba(self, applicant: dict | pd.DataFrame) -> np.ndarray:
        """Return P(default) for one or many applicants."""
        if isinstance(applicant, dict):
            X = pd.DataFrame([applicant])
        else:
            X = applicant
        X_proc = self._prepare(X)
        return self.model.predict_proba(X_proc)[:, 1]

    def predict(self, applicant: dict | pd.DataFrame, threshold: float = 0.5) -> dict:
        """Predict with a verdict and key explanation fields."""
        proba = float(self.predict_proba(applicant)[0]) if isinstance(applicant, dict) else self.predict_proba(applicant)
        if isinstance(applicant, dict):
            verdict = "high_risk" if proba >= threshold else "low_risk"
            tier = self._risk_tier(proba)
            return {
                "probability_of_default": round(proba, 4),
                "verdict": verdict,
                "risk_tier": tier,
                "threshold_used": threshold,
            }
        return {"probabilities": proba.tolist()}

    @staticmethod
    def _risk_tier(p: float) -> str:
        if p < 0.10:
            return "A — Prime"
        if p < 0.20:
            return "B — Near-prime"
        if p < 0.35:
            return "C — Sub-prime"
        if p < 0.55:
            return "D — High risk"
        return "E — Very high risk"
