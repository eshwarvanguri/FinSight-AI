"""Lightweight model & data monitoring utilities.

Tracks:
  - Population Stability Index (PSI) — feature drift
  - Kolmogorov-Smirnov test — distribution drift
  - Prediction distribution shift
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Compute Population Stability Index between two arrays.

    PSI < 0.1: no significant shift
    0.1–0.25:  moderate shift
    > 0.25:    significant shift; retrain
    """
    ref = pd.Series(reference).dropna()
    cur = pd.Series(current).dropna()

    breakpoints = np.quantile(ref, np.linspace(0, 1, bins + 1))
    breakpoints[0] -= 1e-6
    breakpoints[-1] += 1e-6

    ref_counts, _ = np.histogram(ref, bins=breakpoints)
    cur_counts, _ = np.histogram(cur, bins=breakpoints)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_drift(reference: np.ndarray, current: np.ndarray) -> dict:
    """KS test for two samples."""
    stat, p = stats.ks_2samp(reference, current)
    return {"ks_stat": float(stat), "p_value": float(p), "drift": p < 0.05}


def feature_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Per-feature PSI + KS report."""
    rows = []
    for col in feature_cols:
        if col not in reference.columns or col not in current.columns:
            continue
        if not pd.api.types.is_numeric_dtype(reference[col]):
            continue
        psi_val = psi(reference[col].values, current[col].values)
        ks = ks_drift(reference[col].values, current[col].values)
        rows.append({
            "feature": col,
            "psi": round(psi_val, 4),
            "ks_stat": round(ks["ks_stat"], 4),
            "p_value": round(ks["p_value"], 4),
            "alert": "HIGH" if psi_val > 0.25 else ("MEDIUM" if psi_val > 0.1 else "LOW"),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False)
