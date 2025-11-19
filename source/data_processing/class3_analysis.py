"""Class3-level analysis helpers (clustering, spend prep)."""
from __future__ import annotations
from typing import Optional
import pandas as pd
import numpy as np

from .analysis_utils import compute_kmeans_tiers

__all__ = ["cluster_products"]


def cluster_products(df: pd.DataFrame,
                     spend_col: str,
                     k: int = 3,
                     log_transform: bool = True,
                     tier_col: str = "kmeans_tier") -> pd.DataFrame:
    """Cluster products within each Class3 using k-means on a spend/amount column.

    Fallback to simple quantile tiers if k-means fails or insufficient rows.
    Returns a DataFrame with columns: Class3, ProductNumber, amount_eur, kmeans_tier.
    """
    if "Class3" not in df.columns or "ProductNumber" not in df.columns:
        raise KeyError("DataFrame must contain 'Class3' and 'ProductNumber'.")

    work_col = spend_col
    base = (
        df.groupby(["Class3", "ProductNumber"], as_index=False)[work_col]
          .sum()
          .rename(columns={work_col: "amount_eur"})
    )

    out_frames = []
    for c3, g in base.groupby("Class3", as_index=False):
        g = g.copy()
        g["amount_eur"] = pd.to_numeric(g["amount_eur"], errors="coerce")
        g = g[g["amount_eur"].notna() & (g["amount_eur"] > 0)]
        if g.empty:
            continue
        if len(g) < k:
            g[tier_col] = "A"
            out_frames.append(g)
            continue
        g["total_spend"] = g["amount_eur"]
        try:
            g = compute_kmeans_tiers(g, spend_col="total_spend", k=k, log_transform=log_transform, tier_col=tier_col)
        except Exception:
            q = g["amount_eur"].rank(pct=True)
            g[tier_col] = np.where(q <= 1/3, "A", np.where(q <= 2/3, "B", "C"))
        out_frames.append(g.drop(columns=["total_spend"], errors="ignore"))

    if not out_frames:
        return pd.DataFrame(columns=["Class3", "ProductNumber", "amount_eur", tier_col])
    return pd.concat(out_frames, ignore_index=True)
