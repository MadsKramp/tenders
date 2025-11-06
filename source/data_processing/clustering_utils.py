"""
clustering_utils.py
-------------------

Utilities for clustering tender-material products using TWO numeric features:
  1) purchase_amount_eur_total  (sum: purchase_amount_eur_2020..2024)
  2) quantity_sold_total        (sum: quantity_sold_2021..2024)

Data source (EU): params.SUPER_TABLE_FQN
Filters honored (SQL-side): class_2/3/4, brand_name, STEP_CountryOfOrigin, supplier_name_string,
                            item_number, keyword_in_desc; purchase-stop policy.

Designed to be used standalone or by a pipeline (e.g., clustering_pipeline.py).
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

PURCHASE_COLS = [
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

BASE_SELECT_COLS = [
    # identifiers / meta
    "item_number",
    "item_description",
    "class_2", "class_3", "class_4",
    "brand_name",
    "STEP_CountryOfOrigin",
    "supplier_name_string",
    "stop_purchase_ind",
    # signals
] + PURCHASE_COLS + QTY_COLS + ORDERS_COLS


def _fmt_in(values: Sequence[str]) -> str:
    safe = [str(v).replace("'", "\\'") for v in values]
    return "(" + ", ".join([f"'{s.upper()}'" for s in safe]) + ")"


def _where_clause() -> str:
    """Build WHERE clause from super_table_params filters."""
    where = []

    mode = (params.PURCHASE_STOP_MODE or "").strip().lower()
    if mode == "strict":
        where.append("stop_purchase_ind = 'N'")
    elif mode == "lenient":
        where.append("(stop_purchase_ind IS NULL OR UPPER(stop_purchase_ind) != 'Y')")

    def add_eq_in(col: str, values: Optional[Sequence[str]]):
        if values:
            where.append(f"UPPER({col}) IN {_fmt_in(values)}")

    add_eq_in("class_2", params.CLASS2)
    add_eq_in("class_3", params.CLASS3)
    add_eq_in("class_4", params.CLASS4)
    add_eq_in("brand_name", params.BRAND_NAME)
    add_eq_in("STEP_CountryOfOrigin", params.COUNTRY_OF_ORIGIN)
    add_eq_in("supplier_name_string", params.GROUP_SUPPLIER)
    add_eq_in("item_number", params.ITEM_NUMBERS)

    # keyword search in description
    if params.KEYWORDS_IN_DESC:
        likes = []
        for kw in params.KEYWORDS_IN_DESC:
            kw = (kw or "").strip()
            if kw:
                likes.append(f"LOWER(item_description) LIKE '%{kw.lower().replace('%','\\%')}%'")
        if likes:
            where.append("(" + " OR ".join(likes) + ")")

    return ("WHERE " + " AND ".join(where)) if where else ""


def _build_sql(columns: Optional[Sequence[str]] = None, limit: Optional[int] = None) -> str:
    sel_cols = columns or BASE_SELECT_COLS
    select_list = ",\n    ".join(sel_cols)
    sql = f"""
SELECT
    {select_list}
FROM {params.SUPER_TABLE_FQN}
{_where_clause()}
"""
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

def fetch_super_table(
    *,
    columns: Optional[Sequence[str]] = None,
    limit: Optional[int] = params.ROW_LIMIT,
    debug_print_sql: bool = False,
) -> pd.DataFrame:
    """
    Fetch filtered rows from super_table with EU location.
    """
    sql = _build_sql(columns=columns, limit=limit)
    if debug_print_sql:
        print(sql)

    bq = BigQueryConnector()
    try:
        df = bq.query(sql, location=params.BQ_LOCATION)  # if your connector supports it
    except TypeError:
        df = bq.query(sql)

    if df is None or df.empty:
        raise ValueError("No data returned from BigQuery with current filters.")

    df = _coerce_numeric(df, PURCHASE_COLS + QTY_COLS + ORDERS_COLS)
    return df


def prepare_two_feature_df(
    df: pd.DataFrame,
    *,
    min_transactions: Optional[int] = params.MIN_TRANSACTIONS,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Adds aggregate columns and applies basic readiness filtering:
      - purchase_amount_eur_total
      - quantity_sold_total
      - (optional) orders_total >= min_transactions
      - drop rows with both totals == 0
    """
    work = df.copy()

    work["purchase_amount_eur_total"] = _sum_columns(work, PURCHASE_COLS)
    work["quantity_sold_total"] = _sum_columns(work, QTY_COLS)

    if any(c in work.columns for c in ORDERS_COLS):
        work["orders_total"] = _sum_columns(work, ORDERS_COLS)
        if min_transactions:
            work = work.loc[work["orders_total"] >= int(min_transactions)].copy()

    work = work.loc[
        (work["purchase_amount_eur_total"] > 0) | (work["quantity_sold_total"] > 0)
    ].copy()

    feature_cols = ["purchase_amount_eur_total", "quantity_sold_total"]
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
    """
    Attach labels and return (df_clustered, summary).
    Summary includes: n_items and medians of the two features.
    """
    lab_col = f"{method_name.lower()}_cluster"
    dfc = df_features.copy()
    dfc[lab_col] = labels

    # Handle DBSCAN noise (-1) in grouping, keep order by cluster size
    summary = (
        dfc.groupby(lab_col, dropna=False)
           .agg(
               n_items=("item_number", "nunique"),
               purchase_amount_eur_total=("purchase_amount_eur_total", "median"),
               quantity_sold_total=("quantity_sold_total", "median"),
           )
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


def plot_feature_scatter(df_clustered: pd.DataFrame, method_name: str = "kmeans") -> None:
    """
    Simple 2D scatter in original feature space, colored by cluster.
    """
    lab_col = f"{method_name.lower()}_cluster"
    if lab_col not in df_clustered.columns:
        print(f"Cluster column '{lab_col}' not found.")
        return

    x = "purchase_amount_eur_total"
    y = "quantity_sold_total"

    groups = df_clustered.groupby(lab_col)
    plt.figure(figsize=(9, 7))
    for cid, g in groups:
        label = "Noise" if cid == -1 else f"Cluster {cid}"
        plt.scatter(g[x], g[y], s=30, alpha=0.7, label=label)

    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Purchase Amount EUR (total)"); plt.ylabel("Quantity Sold (total)")
    plt.title(f"{method_name.title()} – Feature Scatter (log-log)")
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
