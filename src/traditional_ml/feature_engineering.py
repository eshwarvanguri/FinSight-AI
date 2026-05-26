"""Feature engineering for the credit risk model.

This module turns raw loan/applicant data into model-ready features. It encodes
domain knowledge from credit risk modeling (utilization, DTI bands, credit
history length, etc.) along with standard preprocessing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.logger import log


@dataclass
class FeatureSpec:
    """Container for the feature spec parsed from config."""

    numeric: list[str]
    categorical: list[str]
    derived: list[str]
    target: str


def parse_emp_length(value: Any) -> float:
    """Convert '10+ years' / '< 1 year' / '3 years' -> float."""
    if pd.isna(value):
        return np.nan
    s = str(value).lower().strip()
    if "10+" in s:
        return 10.0
    if "<" in s or "less" in s:
        return 0.5
    digits = "".join(c for c in s if c.isdigit())
    return float(digits) if digits else np.nan


def parse_term(value: Any) -> int:
    """'36 months' -> 36."""
    if pd.isna(value):
        return 36
    digits = "".join(c for c in str(value) if c.isdigit())
    return int(digits) if digits else 36


def parse_pct(value: Any) -> float:
    """'13.49%' -> 13.49 ; '13.49' -> 13.49."""
    if pd.isna(value):
        return np.nan
    s = str(value).replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


class CreditFeatureEngineer:
    """Build derived credit-risk features from raw data."""

    def __init__(self, spec: FeatureSpec):
        self.spec = spec

    def fit(self, df: pd.DataFrame) -> "CreditFeatureEngineer":  # noqa: D401
        """No-op; included for sklearn-style API symmetry."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all derived-feature transformations."""
        df = df.copy()

        # Normalize string-encoded fields
        if "emp_length" in df.columns:
            df["emp_length"] = df["emp_length"].apply(parse_emp_length)
        if "term" in df.columns:
            df["term"] = df["term"].apply(parse_term)
        if "int_rate" in df.columns:
            df["int_rate"] = df["int_rate"].apply(parse_pct)
        if "revol_util" in df.columns:
            df["revol_util"] = df["revol_util"].apply(parse_pct)

        # Derived features (the interesting ones)
        if {"annual_inc", "loan_amnt"}.issubset(df.columns):
            df["income_to_loan_ratio"] = df["annual_inc"] / (df["loan_amnt"] + 1)

        if {"annual_inc", "dti"}.issubset(df.columns):
            # rough monthly debt
            df["monthly_debt"] = (df["annual_inc"] / 12.0) * (df["dti"] / 100.0)
            df["debt_to_income_v2"] = df["monthly_debt"] / (df["annual_inc"] / 12.0 + 1)

        if "revol_util" in df.columns:
            df["utilization_bucket"] = pd.cut(
                df["revol_util"],
                bins=[-0.01, 30, 60, 80, 100, np.inf],
                labels=["low", "moderate", "high", "very_high", "maxed"],
            ).astype(str)

        if {"open_acc", "total_acc"}.issubset(df.columns):
            df["credit_history_length"] = df["total_acc"] - df["open_acc"]
            df["open_account_ratio"] = df["open_acc"] / (df["total_acc"] + 1)

        if {"installment", "annual_inc"}.issubset(df.columns):
            df["installment_burden"] = df["installment"] * 12.0 / (df["annual_inc"] + 1)

        log.info(f"Engineered features. Shape: {df.shape}")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def build_preprocessing_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    scaling: str = "standard",
) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for numeric + categorical features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler() if scaling == "standard" else "passthrough"),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def encode_target(y: pd.Series, positive_class: str = "Charged Off") -> np.ndarray:
    """Binary-encode the target: 1 = bad loan, 0 = good loan."""
    bad_statuses = {positive_class, "Default", "Late (31-120 days)"}
    return y.isin(bad_statuses).astype(int).values
