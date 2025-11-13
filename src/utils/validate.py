from __future__ import annotations
import pandas as pd


def expect_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def expect_non_empty(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("DataFrame is empty")
