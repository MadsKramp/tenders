"""clustering_utils
====================

Updated utilities for clustering tender-material products using a dynamic set of
numeric features derived from yearly columns defined in
``clustering_params.feature_spec()``.

Core philosophy now matches ``clustering_pipeline`` and ``analysis_utils``:
    * Simplified filtering surface (Class2, Class3, Brand, GroupVendorName, description regex)
    * Data fetched via ``analysis_utils.fetch_super_table_filtered`` instead of manual SQL.
    * Feature derivation supports strategies: ``sum`` | ``latest`` | ``mean``.

Backward compatibility:
    * Legacy helpers (manual SQL builder, purchase_stop_ind filters) retained but marked deprecated.
    * Two-feature specific helpers adapted to dynamic feature list.

Primary high-level steps for clustering outside the pipeline:
    1. ``fetch_filtered_super_table()`` – get filtered slice.
    2. ``compute_totals(df)`` (from analysis_utils) – optional wide totals helper.
    3. ``build_feature_columns(df)`` – derive clustering feature columns.
    4. Scale with ``scale_features``.
    5. Optimize / run clustering (KMeans / Agglomerative / DBSCAN).
    6. Summarize with ``analyze_clusters``.

These utilities are import-safe and fall back gracefully if optional pieces are
missing. Visualizations & exports remain similar to previous version.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# -----------------------------------------------------------------------------
# Updated filtered fetch & totals helpers
# -----------------------------------------------------------------------------
try:  # prefer absolute import
    from source.data_processing.analysis_utils import (
        fetch_super_table_filtered,
        compute_totals,
    )
except ImportError:  # pragma: no cover
    try:
        from .analysis_utils import fetch_super_table_filtered, compute_totals  # type: ignore
    except Exception:  # pragma: no cover
        fetch_super_table_filtered = None  # type: ignore
        compute_totals = None  # type: ignore

# -----------------------------------------------------------------------------
# Config / params
# -----------------------------------------------------------------------------
try:
    # Prefer the unified clustering_params module (authoritative configuration)
    from . import clustering_params as params  # type: ignore
except Exception:  # pragma: no cover - fallback if relative import fails
    try:
        import clustering_params as params  # type: ignore
    except Exception as _e:  # final fallback with explicit error
        raise ImportError("Unable to import clustering_params; ensure file 'clustering_params.py' exists in 'source/data_processing'.") from _e

# -----------------------------------------------------------------------------
# BigQuery connector
# (Your file is at: C:\Users\madsh\Data Analysis\tenders-1\db_connect\bigquery_connector.py)
# -----------------------------------------------------------------------------
try:
    from source.db_connect.bigquery_connector import BigQueryConnector
except ImportError:
    from db_connect.bigquery_connector import BigQueryConnector  # type: ignore


# =============================================================================
# Constants & helpers
# =============================================================================

PURCHASE_COLS = [  # retained for backward compatibility / totals
    "purchase_amount_eur_2020",
    "purchase_amount_eur_2021",
    "purchase_amount_eur_2022",
    "purchase_amount_eur_2023",
    "purchase_amount_eur_2024",
]

QTY_COLS = [
    "quantity_sold_2021",
    "quantity_sold_2022",
    "quantity_sold_2023",
    "quantity_sold_2024",
]

ORDERS_COLS = ["orders_2021", "orders_2022", "orders_2023", "orders_2024"]

# Deprecated legacy base column list (manual SQL path). Use fetch_super_table_filtered instead.
BASE_SELECT_COLS = [
    "item_number",
    "item_description",
    "class_2", "class_3", "class_4",  # class_4 kept only for legacy reference
    "brand_name",
    "STEP_CountryOfOrigin",
    "supplier_name_string",
    "stop_purchase_ind",
] + PURCHASE_COLS + QTY_COLS + ORDERS_COLS


def _deprecated_where_clause() -> str:  # pragma: no cover - retained only for legacy
    """Deprecated legacy WHERE clause builder (use filtered_fetch_kwargs + fetch_super_table_filtered)."""
    return ""  # intentionally no-op now


def _deprecated_build_sql(columns: Optional[Sequence[str]] = None, limit: Optional[int] = None) -> str:  # pragma: no cover
    """Deprecated manual SQL construction (replaced by fetch_super_table_filtered)."""
    sel_cols = columns or BASE_SELECT_COLS
    select_list = ",\n    ".join(sel_cols)
    sql = f"SELECT\n    {select_list}\nFROM {params.SUPER_TABLE_FQN}\n"  # legacy path omitted filters
    if limit and isinstance(limit, int) and limit > 0:
        sql += f"LIMIT {limit}\n"
    return sql


def _sum_columns(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    cols_present = [c for c in cols if c in df.columns]
    if not cols_present:
        # nothing to sum -> 0
        return pd.Series(0, index=df.index, dtype=float)
    return df.loc[:, cols_present].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)


def _coerce_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# =============================================================================
# Data fetch & feature preparation
# =============================================================================

def fetch_filtered_super_table(*, include_purchase: bool = True, include_description_regex: bool = True) -> pd.DataFrame:
    """Public fetch wrapper using simplified filtering API.

    Parameters mirror ``clustering_params.filtered_fetch_kwargs`` (those are injected automatically).
    Additional toggles allow ignoring description regex or purchase_data join if needed.
    """
    if fetch_super_table_filtered is None:
        raise ImportError("analysis_utils.fetch_super_table_filtered not available; ensure analysis_utils.py is present.")
    kwargs = params.filtered_fetch_kwargs()
    if not include_description_regex:
        kwargs["description_regex"] = None
    if not include_purchase:
        kwargs["include_purchase"] = False
    df = fetch_super_table_filtered(**kwargs)
    if df is None or df.empty:
        raise ValueError("No data returned with current simplified filters.")
    # Totals helper (optional if compute_totals available)
    if compute_totals is not None:
        df = compute_totals(df)
    return df


def build_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive feature columns specified in params.feature_spec().

    Each spec entry: {'from': [...year cols...], 'strategy': 'sum|latest|mean'}.
    Missing source columns -> feature filled with 0.0.
    """
    spec = params.feature_spec()
    work = df.copy()
    for feat_name, cfg in spec.items():
        cols = [c for c in cfg.get("from", []) if c in work.columns]
        if not cols:
            work[feat_name] = 0.0
            continue
        block = work[cols].apply(pd.to_numeric, errors="coerce")
        strat = str(cfg.get("strategy", "sum")).lower()
        if strat == "latest":
            work[feat_name] = block.ffill(axis=1).iloc[:, -1].fillna(0)
        elif strat == "mean":
            work[feat_name] = block.mean(axis=1).fillna(0)
        else:
            work[feat_name] = block.sum(axis=1).fillna(0)
    return work


def prepare_feature_df(
    df: pd.DataFrame,
    *,
    min_transactions: Optional[int] = params.MIN_TRANSACTIONS,
) -> Tuple[pd.DataFrame, List[str]]:
    """Apply feature derivation + basic filters.

    - Derive dynamic feature columns per feature_spec.
    - Optional transaction threshold (orders_total) if present.
    - Drop rows where all features == 0.
    Returns (prepared_df, feature_column_list).
    """
    work = build_feature_columns(df)

    # Transaction threshold using orders_total if available
    if "orders_total" in work.columns and min_transactions:
        work = work.loc[work["orders_total"] >= int(min_transactions)].copy()

    feature_cols = list(params.CLUSTER_FEATURES)
    # Remove all-zero rows across features
    zero_mask = (work[feature_cols].astype(float).fillna(0) == 0).all(axis=1)
    work = work.loc[~zero_mask].copy()
    return work, feature_cols


def scale_features(df: pd.DataFrame, feature_columns: List[str]) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardize features with StandardScaler.
    """
    scaler = StandardScaler()
    X = df[feature_columns].astype(float).fillna(0).values
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# =============================================================================
# Cluster optimization & methods
# =============================================================================

def find_optimal_clusters(
    X: np.ndarray,
    *,
    max_clusters: int = params.MAX_CLUSTERS,
    min_clusters: int = params.MIN_CLUSTERS,
    random_state: int = params.RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Find K (for KMeans/Agglomerative) using silhouette, Calinski-Harabasz, and a simple inertia elbow.
    """
    n_min = max(2, int(min_clusters))
    n_max = max(n_min, int(max_clusters))
    n_max = min(n_max, max(2, X.shape[0] - 1))

    inertia: Dict[int, float] = {}
    sil: Dict[int, float] = {}
    ch: Dict[int, float] = {}

    for k in range(n_min, n_max + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = km.fit_predict(X)
        inertia[k] = float(km.inertia_)
        try:
            sil[k] = float(silhouette_score(X, labels))
        except Exception:
            sil[k] = float("nan")
        try:
            ch[k] = float(calinski_harabasz_score(X, labels))
        except Exception:
            ch[k] = float("nan")

    ks = sorted(inertia.keys())
    # Elbow: largest relative drop
    rel_drops = {}
    for i in range(1, len(ks)):
        k_prev, k_curr = ks[i - 1], ks[i]
        prev, curr = inertia[k_prev], inertia[k_curr]
        rel_drops[k_curr] = (prev - curr) / prev if prev else 0.0
    elbow_k = max(rel_drops, key=rel_drops.get) if rel_drops else ks[0]

    best_sil_k = max((k for k in ks if not math.isnan(sil[k])), key=lambda k: sil[k], default=ks[0])
    best_ch_k = max((k for k in ks if not math.isnan(ch[k])), key=lambda k: ch[k], default=ks[0])

    return dict(
        inertia=inertia,
        silhouette=sil,
        calinski=ch,
        cluster_range=ks,
        optimal_elbow=elbow_k,
        optimal_silhouette=best_sil_k,
        optimal_calinski=best_ch_k,
    )


def perform_kmeans_clustering(
    X: np.ndarray, n_clusters: int, *, random_state: int = params.RANDOM_STATE
) -> Tuple[np.ndarray, Any]:
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    labels = km.fit_predict(X)
    return labels, km


def perform_hierarchical_clustering(X: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, Any]:
    ag = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = ag.fit_predict(X)
    return labels, ag


def _auto_tune_dbscan(X: np.ndarray) -> Tuple[float, int]:
    """
    Heuristic for DBSCAN params: min_samples ~ log-base-1.5(N), eps ~ 95th pct of kth neighbor distance.
    """
    n = max(2, X.shape[0])
    min_samples = max(5, int(round(math.log(n, 1.5))))

    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min_samples)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        kth = distances[:, -1]
        eps = float(np.quantile(kth, 0.95))
    except Exception:
        eps = 0.5  # fallback

    return eps, min_samples


def perform_dbscan_clustering(
    X: np.ndarray, *, eps: Optional[float] = None, min_samples: Optional[int] = None
) -> Tuple[np.ndarray, Any]:
    if eps is None or min_samples is None:
        auto_eps, auto_min = _auto_tune_dbscan(X)
        eps = eps or auto_eps
        min_samples = min_samples or auto_min

    db = DBSCAN(eps=float(eps), min_samples=int(min_samples))
    labels = db.fit_predict(X)
    return labels, db


# =============================================================================
# Analysis & summaries
# =============================================================================

def analyze_clusters(
    df_features: pd.DataFrame,
    labels: np.ndarray,
    feature_columns: List[str],
    *,
    method_name: str = "kmeans",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Attach labels and produce summary (n_items + median & mean for each feature).

    Works with arbitrary number of features; DBSCAN noise (-1) retained.
    """
    lab_col = f"{method_name.lower()}_cluster"
    dfc = df_features.copy()
    dfc[lab_col] = labels

    agg_map = {"n_items": ("item_number", "nunique")}
    for c in feature_columns:
        agg_map[f"median_{c}"] = (c, "median")
        agg_map[f"mean_{c}"] = (c, "mean")

    summary = (
        dfc.groupby(lab_col, dropna=False)
           .agg(**agg_map)
           .reset_index()
           .sort_values("n_items", ascending=False)
    )
    return dfc, summary


# =============================================================================
# Plotting helpers (matplotlib only)
# =============================================================================

def plot_cluster_optimization_metrics(optimization: Dict[str, Any]) -> None:
    ks = optimization["cluster_range"]
    sil = [optimization["silhouette"].get(k, np.nan) for k in ks]
    ch = [optimization["calinski"].get(k, np.nan) for k in ks]
    inertia = [optimization["inertia"].get(k, np.nan) for k in ks]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].plot(ks, sil, marker="o")
    ax[0].axvline(optimization["optimal_silhouette"], ls="--")
    ax[0].set_title("Silhouette vs K"); ax[0].set_xlabel("K"); ax[0].set_ylabel("Silhouette"); ax[0].grid(True, alpha=.3)

    ax[1].plot(ks, ch, marker="o")
    ax[1].axvline(optimization["optimal_calinski"], ls="--")
    ax[1].set_title("Calinski–Harabasz vs K"); ax[1].set_xlabel("K"); ax[1].grid(True, alpha=.3)

    ax[2].plot(ks, inertia, marker="o")
    ax[2].axvline(optimization["optimal_elbow"], ls="--")
    ax[2].set_title("Inertia vs K (Elbow)"); ax[2].set_xlabel("K"); ax[2].grid(True, alpha=.3)

    plt.tight_layout(); plt.show()


def plot_feature_scatter(
    df_clustered: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    method_name: str = "kmeans",
    log_scale: bool = True,
) -> None:
    """Scatter plot for first two feature columns.

    If less than 2 features, exits gracefully. Optionally apply log scaling.
    """
    lab_col = f"{method_name.lower()}_cluster"
    if lab_col not in df_clustered.columns:
        print(f"Cluster column '{lab_col}' not found; cannot plot.")
        return
    feature_columns = feature_columns or [c for c in params.CLUSTER_FEATURES]
    if len(feature_columns) < 2:
        print("Need at least two features to scatter plot.")
        return
    x, y = feature_columns[:2]
    groups = df_clustered.groupby(lab_col)
    plt.figure(figsize=(9, 7))
    for cid, g in groups:
        label = "Noise" if cid == -1 else f"Cluster {cid}"
        plt.scatter(g[x], g[y], s=30, alpha=0.7, label=label)
    if log_scale:
        plt.xscale("log"); plt.yscale("log")
    plt.xlabel(x); plt.ylabel(y)
    plt.title(f"{method_name.title()} – Feature Scatter")
    plt.grid(True, which="both", ls=":", alpha=.3)
    plt.legend()
    plt.tight_layout(); plt.show()


# =============================================================================
# Export helpers
# =============================================================================

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _excel_name(stem: str, folder: str = params.OUTPUT_DIR) -> str:
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{stem}_{_timestamp()}.xlsx")


def save_clustering_results(
    df_clustered: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    *,
    rounding_filter: Optional[float] = params.ROUNDING_FILTER,
) -> Tuple[str, str]:
    """
    Save detailed & summary results to Excel (timestamped). Returns file paths.
    """
    suffix = f"_rounding_{rounding_filter}" if rounding_filter is not None else "_all_roundings"

    detailed_path = _excel_name(f"tender_clusters_detailed{suffix}")
    summary_path = _excel_name(f"tender_clusters_summary{suffix}")

    df_clustered.to_excel(detailed_path, index=False)
    cluster_summary.to_excel(summary_path, index=False)

    print(f"💾 Saved: {detailed_path}")
    print(f"💾 Saved: {summary_path}")
    return detailed_path, summary_path


def save_clustering_results_bq(
    df_item_cluster: pd.DataFrame,
    method: str,
    rounding_filter: Optional[float] = params.ROUNDING_FILTER,
    class3: Optional[str] = None,
    product_description: Optional[str] = None,
    *,
    dataset_id: str = params.DATASET_ID,
    table_id: str = "tender_material_clusters",
) -> None:
    """
    Persist (item_number, cluster_id, metadata) to BigQuery.
    Expects df_item_cluster with columns: item_number, cluster_id
    """
    need = {"item_number", "cluster_id"}
    if not need.issubset(df_item_cluster.columns):
        raise ValueError("df_item_cluster must contain columns {'item_number','cluster_id'}.")

    out = df_item_cluster.copy()
    out["cluster_method"] = str(method).lower()
    out["rounding_filter"] = rounding_filter
    out["class3_filter"] = class3
    out["description_filter"] = product_description
    out["batch_ts"] = datetime.utcnow()

    print(f"Saving {len(out)} rows to BQ: {params.PROJECT_ID}.{dataset_id}.{table_id} (EU)")
    bq = BigQueryConnector()
    try:
        bq.save_to_table(df=out, dataset_id=dataset_id, table_id=table_id, project_id=params.PROJECT_ID)  # type: ignore
    except TypeError:
        # If your connector signature differs, adapt here:
        bq.save_to_table(df=out, dataset_id=dataset_id, table_id=table_id)  # type: ignore
