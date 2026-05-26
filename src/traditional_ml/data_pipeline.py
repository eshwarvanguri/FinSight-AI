"""Data loading and splitting pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import log


def load_data(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load a CSV/Parquet file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")

    log.info(f"Loading data from {p}")
    if p.suffix == ".csv":
        df = pd.read_csv(p, nrows=nrows, low_memory=False)
    elif p.suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
        if nrows:
            df = df.head(nrows)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")

    log.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


def clean_data(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Drop rows with no target, drop columns with > 70% nulls."""
    initial_rows = len(df)
    df = df.dropna(subset=[target_col])
    log.info(f"Dropped {initial_rows - len(df)} rows with missing target")

    null_pct = df.isnull().mean()
    too_sparse = null_pct[null_pct > 0.7].index.tolist()
    if too_sparse:
        log.warning(f"Dropping {len(too_sparse)} columns with > 70% nulls: {too_sparse[:5]}...")
        df = df.drop(columns=too_sparse)

    df = df.drop_duplicates()
    return df


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    stratify: bool = True,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Train/val/test split with stratification."""
    stratify_col = df[target_col] if stratify else None

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=stratify_col,
        random_state=random_state,
    )

    relative_val = val_size / (1 - test_size)
    stratify_tv = train_val[target_col] if stratify else None
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=stratify_tv,
        random_state=random_state,
    )

    log.info(
        f"Split sizes — train: {len(train):,}, val: {len(val):,}, test: {len(test):,}"
    )

    return {"train": train, "val": val, "test": test}


def save_splits(splits: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    """Persist splits as parquet."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        path = out / f"{name}.parquet"
        df.to_parquet(path, index=False)
        log.info(f"Saved {name} → {path}")
