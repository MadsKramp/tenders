"""ABC / Pareto Spend Tier Segmentation.

Provides business-friendly spend tiering (A/B/C) based on cumulative share
of total spend plus optional model-driven cutpoints using 1-D clustering on
log-transformed spend values.

Primary method: compute_abc_tiers
Optional methods: compute_kmeans_tiers, compute_gmm_tiers

Assumptions:
 - Input DataFrame contains a per-product spend metric column (default 'total_spend').
 - If absent, will attempt to derive from 'purchase_amount_eur' * 'purchase_quantity'.
 - Thresholds follow standard Pareto logic (e.g., 80% / 95%) but are configurable.

Note: Model-driven tiers are complementary; business tiering should remain
stable and interpretable for sourcing strategy alignment.
"""

from __future__ import annotations

from typing import Tuple, Optional, Dict
import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
except Exception:  # sklearn might not be present in minimal env
    KMeans = None
    GaussianMixture = None

ABC_DEFAULT_THRESHOLDS: Tuple[float, float] = (0.80, 0.95)


def _resolve_spend_column(df: pd.DataFrame, spend_col: str) -> Tuple[pd.DataFrame, str]:
    """Ensure a spend column exists; derive if necessary.

    Derivation path:
      - If spend_col not present but purchase_amount_eur & purchase_quantity present,
        compute spend_col = purchase_amount_eur * purchase_quantity.
    """
    if spend_col not in df.columns:
        if {'purchase_amount_eur', 'purchase_quantity'} <= set(df.columns):
            derived = df['purchase_amount_eur'] * df['purchase_quantity']
            df = df.copy()
            df[spend_col] = derived
            print(f"ℹ️ Derived '{spend_col}' from purchase_amount_eur * purchase_quantity.")
        else:
            raise ValueError(
                f"Spend column '{spend_col}' not found and cannot derive (missing purchase_amount_eur/purchase_quantity)."
            )
    return df, spend_col


def compute_abc_tiers(
    df: pd.DataFrame,
    spend_col: str = 'total_spend',
    thresholds: Tuple[float, float] = ABC_DEFAULT_THRESHOLDS,
    ascending: bool = False,
    tier_col: str = 'abc_tier'
) -> pd.DataFrame:
    """Assign A/B/C tiers based on cumulative share of total spend.

    Args:
        df: Product-level DataFrame.
        spend_col: Column containing total spend over analysis window.
        thresholds: (A_cut, B_cut) cumulative share thresholds. A <= A_cut, B <= B_cut, else C.
        ascending: If True, treats low spend as A (usually False).
        tier_col: Output column name for tiers.
    Returns:
        DataFrame with tier_col added.
    """
    if len(thresholds) != 2 or not (0 < thresholds[0] < thresholds[1] < 1):
        raise ValueError("Thresholds must be a tuple (a_cut, b_cut) with 0 < a_cut < b_cut < 1.")
    df, spend_col = _resolve_spend_column(df, spend_col)
    work = df[[spend_col]].copy()
    work['spend'] = pd.to_numeric(work[spend_col], errors='coerce').fillna(0)
    sort_order = work['spend'].sort_values(ascending=ascending)
    ranked = work.loc[sort_order.index].copy()
    total = ranked['spend'].sum()
    if total <= 0:
        raise ValueError("Total spend is zero; cannot compute ABC tiers.")
    ranked['cum_share'] = ranked['spend'].cumsum() / total
    a_cut, b_cut = thresholds
    def assign_tier(cum: float) -> str:
        if cum <= a_cut:
            return 'A'
        if cum <= b_cut:
            return 'B'
        return 'C'
    ranked[tier_col] = ranked['cum_share'].apply(assign_tier)
    # Map back to original df order
    out = df.copy()
    out[tier_col] = ranked[tier_col].reindex(out.index)
    return out


def compute_kmeans_tiers(
    df: pd.DataFrame,
    spend_col: str = 'total_spend',
    k: int = 3,
    log_transform: bool = True,
    tier_col: str = 'kmeans_tier'
) -> pd.DataFrame:
    """Assign tiers using 1-D k-means clustering on log(spend).

    Clusters are ordered by mean spend descending mapped to A,B,C (truncate if k<3).
    """
    if KMeans is None:
        raise ImportError("sklearn not available for k-means tiering.")
    if k < 2 or k > 6:
        raise ValueError("k should be between 2 and 6 for interpretability.")
    df, spend_col = _resolve_spend_column(df, spend_col)
    spend_vals = pd.to_numeric(df[spend_col], errors='coerce').fillna(0).values.reshape(-1, 1)
    X = np.log1p(spend_vals) if log_transform else spend_vals
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_.flatten()
    order = np.argsort(centers)[::-1]  # descending spend
    tier_map = {}
    abc_seq = list('ABCDEF')
    for idx, cluster_id in enumerate(order):
        tier_map[cluster_id] = abc_seq[idx]
    out = df.copy()
    out[tier_col] = [tier_map[l] for l in labels]
    return out


def compute_gmm_tiers(
    df: pd.DataFrame,
    spend_col: str = 'total_spend',
    max_components: int = 3,
    log_transform: bool = True,
    tier_col: str = 'gmm_tier'
) -> pd.DataFrame:
    """Assign tiers using 1-D Gaussian Mixture selecting component count by BIC.

    Components ordered by mean spend descending -> A,B,C.
    """
    if GaussianMixture is None:
        raise ImportError("sklearn not available for GMM tiering.")
    if max_components < 2 or max_components > 6:
        raise ValueError("max_components should be between 2 and 6.")
    df, spend_col = _resolve_spend_column(df, spend_col)
    spend = pd.to_numeric(df[spend_col], errors='coerce').fillna(0).values
    X = np.log1p(spend).reshape(-1, 1) if log_transform else spend.reshape(-1, 1)
    best_gmm = None
    best_bic = np.inf
    for n in range(2, max_components + 1):
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(X)
        bic = gmm.bic(X)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
    labels = best_gmm.predict(X)
    means = best_gmm.means_.flatten()
    order = np.argsort(means)[::-1]
    tier_map: Dict[int, str] = {}
    abc_seq = list('ABCDEF')
    for idx, comp_id in enumerate(order):
        tier_map[comp_id] = abc_seq[idx]
    out = df.copy()
    out[tier_col] = [tier_map[l] for l in labels]
    return out


def summarize_tiers(df: pd.DataFrame, spend_col: str, tier_col: str) -> pd.DataFrame:
    """Return summary (count, spend, share) per tier."""
    df, spend_col = _resolve_spend_column(df, spend_col)
    total = pd.to_numeric(df[spend_col], errors='coerce').fillna(0).sum()
    grp = df.groupby(tier_col)[spend_col].agg(['count', 'sum']).rename(columns={'sum': 'spend'})
    grp['share_pct'] = grp['spend'] / total * 100.0 if total else 0.0
    return grp.reset_index().sort_values('share_pct', ascending=False)


__all__ = [
    'compute_abc_tiers',
    'compute_kmeans_tiers',
    'compute_gmm_tiers',
    'summarize_tiers'
]
