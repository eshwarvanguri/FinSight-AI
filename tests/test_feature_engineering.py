"""Tests for feature engineering."""
import numpy as np
import pandas as pd
import pytest

from src.traditional_ml.feature_engineering import (
    CreditFeatureEngineer,
    FeatureSpec,
    encode_target,
    parse_emp_length,
    parse_pct,
    parse_term,
)


def test_parse_emp_length():
    assert parse_emp_length("10+ years") == 10.0
    assert parse_emp_length("< 1 year") == 0.5
    assert parse_emp_length("3 years") == 3.0
    assert np.isnan(parse_emp_length(None))


def test_parse_term():
    assert parse_term("36 months") == 36
    assert parse_term("60 months") == 60
    assert parse_term(None) == 36


def test_parse_pct():
    assert parse_pct("13.49%") == 13.49
    assert parse_pct("13.49") == 13.49
    assert np.isnan(parse_pct("abc"))


def test_feature_engineer_basic():
    spec = FeatureSpec(numeric=[], categorical=[], derived=[], target="loan_status")
    fe = CreditFeatureEngineer(spec)
    df = pd.DataFrame({
        "annual_inc": [60000, 100000],
        "loan_amnt": [10000, 30000],
        "dti": [15.0, 25.0],
        "revol_util": [40.0, 90.0],
        "open_acc": [5, 10],
        "total_acc": [20, 30],
        "installment": [300, 600],
        "emp_length": ["3 years", "10+ years"],
        "term": ["36 months", "60 months"],
        "int_rate": ["10.5%", "15.0%"],
        "loan_status": ["Fully Paid", "Charged Off"],
    })
    out = fe.transform(df)
    assert "income_to_loan_ratio" in out.columns
    assert "utilization_bucket" in out.columns
    assert out["emp_length"].iloc[1] == 10.0
    assert out["term"].iloc[0] == 36


def test_encode_target():
    s = pd.Series(["Fully Paid", "Charged Off", "Default", "Current"])
    y = encode_target(s)
    assert list(y) == [0, 1, 1, 0]
