"""
clustering_pipeline.py
----------------------

Scalable clustering pipeline on tender material using TWO features:
- purchase_amount_eur_total  (sum of purchase_amount_eur_2020..2024)
- quantity_sold_total        (sum of quantity_sold_2021..2024)

Data source (EU):
  kramp-sharedmasterdata-prd.MadsH.super_table

Filters (SQL-side) respected:
  class_2 / class_3 / class_4 / brand_name / STEP_CountryOfOrigin / supplier_name_string
  purchase stop mode: 'strict' (stop_purchase_ind='N'), 'lenient' (!='Y' or null), or None

Clustering methods:
  - KMeans
  - Agglomerative (hierarchical)
  - DBSCAN (auto tunes eps/min_samples if not provided)

Usage (example):
    from source.clustering_pipeline import ClusteringPipeline

    pipe = ClusteringPipeline()
    pipe.run_complete_analysis(include_dbscan=True, show_visualizations=False)
    df_clusters = pipe.clustering_results["kmeans"]["df_clustered"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# --- params import (single, authoritative) ---
try:
    from source.data_processing import clustering_params as params  # absolute within project
except ImportError:
    try:
        from . import clustering_params as params  # relative fallback
    except ImportError:
        import clustering_params as params  # type: ignore  # last-resort flat script

# --- BigQuery connector (canonical absolute import) ---
try:
    from source.db_connect.bigquery_connector import BigQueryConnector
except ImportError:
    try:
        from ..db_connect.bigquery_connector import BigQueryConnector  # type: ignore
    except Exception:
        from db_connect.bigquery_connector import BigQueryConnector  # type: ignore


# =============================================================================
# Helpers - SQL building & data munging
# =============================================================================

_PURCHASE_COLS = [
    "purchase_amount_eur_2020",
    "purchase_amount_eur_2021",
    "purchase_amount_eur_2022",
    "purchase_amount_eur_2023",
    "purchase_amount_eur_2024",
]

_QTY_COLS = [
    "quantity_sold_2021",
    "quantity_sold_2022",
    "quantity_sold_2023",
    "quantity_sold_2024",
]

_ORDERS_COLS = [
    "orders_2021", "orders_2022", "orders_2023", "orders_2024"
]

_BASE_SELECT_COLS = [
    "item_number",
    "item_description",
    "class_2", "class_3", "class_4",
    "brand_name",
    "STEP_CountryOfOrigin",
    "supplier_name_string",
    "stop_purchase_ind",
] + _PURCHASE_COLS + _QTY_COLS + _ORDERS_COLS


def _fmt_in(values: Sequence[str]) -> str:
    # Case-insensitive IN with UPPER() on both sides
    safe = [v.replace("'", "\\'") for v in values]
    return "(" + ", ".join([f"'{s.upper()}'" for s in safe]) + ")"


def _where_clause() -> str:
    """Compose WHERE clause from params.* filters + purchase stop mode."""
    where = []

    # Purchase stop policy
    mode = (params.PURCHASE_STOP_MODE or "").strip().lower()
    if mode == "strict":
        where.append("stop_purchase_ind = 'N'")
    elif mode == "lenient":
        where.append("(stop_purchase_ind IS NULL OR UPPER(stop_purchase_ind) != 'Y')")
    # None -> no purchase-stop filter

    def add_filter(col: str, values: Optional[Sequence[str]]):
        if values:
            upper_vals = _fmt_in(values)
            where.append(f"UPPER({col}) IN {upper_vals}")

    add_filter("class_2", params.CLASS2)
    add_filter("class_3", params.CLASS3)
    add_filter("class_4", params.CLASS4)
    add_filter("brand_name", params.BRAND_NAME)
    add_filter("STEP_CountryOfOrigin", params.COUNTRY_OF_ORIGIN)
    add_filter("supplier_name_string", params.GROUP_SUPPLIER)
    add_filter("item_number", params.ITEM_NUMBERS)

    # Keyword search in description (case-insensitive LIKE)
    if params.KEYWORDS_IN_DESC:
        likes = []
        for kw in params.KEYWORDS_IN_DESC:
            kw = kw.strip()
            if kw:
                likes.append(f"LOWER(item_description) LIKE '%{kw.lower().replace('%','\\%')}%'")
        if likes:
            where.append("(" + " OR ".join(likes) + ")")

    return ("WHERE " + " AND ".join(where)) if where else ""


def _build_sql(columns: Optional[Sequence[str]] = None, limit: Optional[int] = None) -> str:
    sel_cols = columns or _BASE_SELECT_COLS
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
    return df.loc[:, [c for c in cols if c in df.columns]].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)


def _coerce_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# =============================================================================
# Core Pipeline
# =============================================================================

@dataclass
class ClusteringPipeline:
    """
    Clusters super-table products by two features:
      - purchase_amount_eur_total
      - quantity_sold_total
    with optional filters and EU BigQuery execution.

    Attributes filled after running:
      - data           : raw filtered DataFrame (selected columns)
      - features       : DataFrame with feature columns + item metadata
      - feature_cols   : ['purchase_amount_eur_total', 'quantity_sold_total']
      - X_scaled       : standardized feature matrix
      - optimization   : metrics/results for cluster-number selection
      - clustering_results: dict of results per method ('kmeans','hierarchical','dbscan')
    """

    # Pull defaults from params
    bq_location: str = params.BQ_LOCATION
    row_limit: Optional[int] = params.ROW_LIMIT
    random_state: int = params.RANDOM_STATE

    # cluster knobs
    min_clusters: int = params.MIN_CLUSTERS
    max_clusters: int = params.MAX_CLUSTERS
    n_clusters_override: Optional[int] = params.N_CLUSTERS_OVERRIDE

    include_kmeans: bool = params.INCLUDE_KMEANS
    include_hierarchical: bool = params.INCLUDE_HIERARCHICAL
    include_dbscan: bool = params.INCLUDE_DBSCAN

    dbscan_eps: Optional[float] = params.DBSCAN_EPS
    dbscan_min_samples: Optional[int] = params.DBSCAN_MIN_SAMPLES

    # internal state
    data: pd.DataFrame | None = None
    features: pd.DataFrame | None = None
    feature_cols: List[str] = field(default_factory=lambda: ["purchase_amount_eur_total", "quantity_sold_total"])
    X_scaled: np.ndarray | None = None
    scaler: StandardScaler | None = None
    optimization: Dict[str, Any] = field(default_factory=dict)
    clustering_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # --------------------------
    # Step 1: Load filtered data
    # --------------------------
    def load_data(self, *, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
        sql = _build_sql(columns=columns, limit=self.row_limit)
        bq = BigQueryConnector()
        try:
            df = bq.query(sql, location=self.bq_location)  # many connectors accept 'location'
        except TypeError:
            df = bq.query(sql)  # fallback if your query() doesn't accept location

        if df is None or df.empty:
            raise ValueError("No data returned from BigQuery with current filters.")

        # Force numeric on year columns we need
        df = _coerce_numeric(df, _PURCHASE_COLS + _QTY_COLS + _ORDERS_COLS)
        self.data = df
        return df

    # -----------------------------------
    # Step 2: Prepare the two clustering features
    # -----------------------------------
    def prepare_features(self, *, min_transactions: Optional[int] = params.MIN_TRANSACTIONS,
                         rounding_filter: Optional[float] = params.ROUNDING_FILTER) -> Tuple[pd.DataFrame, List[str]]:
        if self.data is None:
            raise RuntimeError("Call load_data() first.")

        df = self.data.copy()

        # Build totals
        df["purchase_amount_eur_total"] = _sum_columns(df, _PURCHASE_COLS)
        df["quantity_sold_total"] = _sum_columns(df, _QTY_COLS)

        # Optional: filter by min total orders (proxy for transactions)
        if min_transactions and any(c in df.columns for c in _ORDERS_COLS):
            df["orders_total"] = _sum_columns(df, _ORDERS_COLS)
            df = df.loc[df["orders_total"] >= int(min_transactions)].copy()

        # Optional: rounding filter exists in sales tables; your super table may not carry it
        # Keeping the knob for compatibility; no-op here unless you have a rounding column joined later.

        # Keep only rows with positive signal in either feature (avoid all-zero rows)
        df = df.loc[(df["purchase_amount_eur_total"] > 0) | (df["quantity_sold_total"] > 0)].copy()

        # Scale features
        feats = df[self.feature_cols].astype(float).fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feats.values)

        self.features = df
        self.X_scaled = X_scaled
        self.scaler = scaler
        return df, self.feature_cols

    # -------------------------------------------------
    # Step 3: Find optimal number of clusters (K/Agglo)
    # -------------------------------------------------
    def optimize_clusters(self) -> Dict[str, Any]:
        if self.X_scaled is None or self.features is None:
            raise RuntimeError("Call prepare_features() first.")

        X = self.X_scaled
        n_min = max(2, int(self.min_clusters))
        n_max = min(max(n_min, int(self.max_clusters)), max(2, X.shape[0] - 1))

        inertia_scores: Dict[int, float] = {}
        sil_scores: Dict[int, float] = {}
        calinski_scores: Dict[int, float] = {}

        for k in range(n_min, n_max + 1):
            km = KMeans(n_clusters=k, n_init="auto", random_state=self.random_state)
            labels = km.fit_predict(X)

            inertia_scores[k] = float(km.inertia_)
            # Some metrics can fail on degenerate labelings
            try:
                sil_scores[k] = float(silhouette_score(X, labels))
            except Exception:
                sil_scores[k] = float("nan")
            try:
                calinski_scores[k] = float(calinski_harabasz_score(X, labels))
            except Exception:
                calinski_scores[k] = float("nan")

        # "Elbow": pick largest relative drop in inertia
        ks = sorted(inertia_scores.keys())
        rel_drops = {}
        for i in range(1, len(ks)):
            k_prev, k_curr = ks[i - 1], ks[i]
            prev, curr = inertia_scores[k_prev], inertia_scores[k_curr]
            rel_drops[k_curr] = (prev - curr) / prev if prev else 0.0
        elbow_k = max(rel_drops, key=rel_drops.get) if rel_drops else ks[0]

        # Silhouette & Calinski best k
        best_sil_k = max((k for k in ks if not math.isnan(sil_scores[k])), key=lambda k: sil_scores[k], default=ks[0])
        best_cal_k = max((k for k in ks if not math.isnan(calinski_scores[k])), key=lambda k: calinski_scores[k], default=ks[0])

        out = dict(
            inertia=inertia_scores,
            silhouette=sil_scores,
            calinski=calinski_scores,
            optimal_elbow=elbow_k,
            optimal_silhouette=best_sil_k,
            optimal_calinski=best_cal_k,
        )
        self.optimization = out
        return out

    # -------------------------------------------------
    # Step 4: Clustering methods
    # -------------------------------------------------
    def _choose_k(self) -> int:
        if self.n_clusters_override:
            return int(self.n_clusters_override)
        if not self.optimization:
            raise RuntimeError("Call optimize_clusters() first or set n_clusters_override.")
        # Majority vote; tie -> pick the greatest
        candidates = [
            self.optimization.get("optimal_silhouette"),
            self.optimization.get("optimal_calinski"),
            self.optimization.get("optimal_elbow"),
        ]
        counts = pd.Series(candidates).value_counts()
        return int(counts.index.max())

    def run_kmeans(self, n_clusters: Optional[int] = None) -> Dict[str, Any]:
        if self.X_scaled is None or self.features is None:
            raise RuntimeError("Call prepare_features() first.")
        k = n_clusters or self._choose_k()
        km = KMeans(n_clusters=k, n_init="auto", random_state=self.random_state)
        labels = km.fit_predict(self.X_scaled)

        res = self._analyze_result(labels, method="kmeans", model=km, n_clusters=k)
        self.clustering_results["kmeans"] = res
        return res

    def run_hierarchical(self, n_clusters: Optional[int] = None) -> Dict[str, Any]:
        if self.X_scaled is None or self.features is None:
            raise RuntimeError("Call prepare_features() first.")
        k = n_clusters or self._choose_k()
        ag = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = ag.fit_predict(self.X_scaled)

        res = self._analyze_result(labels, method="hierarchical", model=ag, n_clusters=k)
        self.clustering_results["hierarchical"] = res
        return res

    def _auto_tune_dbscan(self) -> Tuple[float, int]:
        # Simple heuristic: min_samples = max(5, round(log(N)))
        n = self.X_scaled.shape[0]
        min_samples = self.dbscan_min_samples or max(5, int(round(math.log(max(n, 2), 1.5))))
        # eps: 95th percentile of distances to the min_samples-th neighbor
        try:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min_samples)
            nn.fit(self.X_scaled)
            distances, _ = nn.kneighbors(self.X_scaled)
            kth = distances[:, -1]
            eps = float(np.quantile(kth, 0.95))
        except Exception:
            eps = 0.5  # fallback
        return eps, min_samples

    def run_dbscan(self, eps: Optional[float] = None, min_samples: Optional[int] = None) -> Dict[str, Any]:
        if self.X_scaled is None or self.features is None:
            raise RuntimeError("Call prepare_features() first.")
        if eps is None or min_samples is None:
            auto_eps, auto_min = self._auto_tune_dbscan()
            eps = eps or auto_eps
            min_samples = min_samples or auto_min

        db = DBSCAN(eps=float(eps), min_samples=int(min_samples))
        labels = db.fit_predict(self.X_scaled)

        # Count clusters ignoring noise (-1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        res = self._analyze_result(
            labels, method="dbscan", model=db, n_clusters=n_clusters,
            extra=dict(
                eps=float(eps),
                min_samples=int(min_samples),
                n_noise=int(np.sum(labels == -1)),
                noise_ratio=float(np.mean(labels == -1)),
            ),
        )
        self.clustering_results["dbscan"] = res
        return res

    # -------------------------------------------------
    # Common post-processing / metrics
    # -------------------------------------------------
    def _analyze_result(self, labels: np.ndarray, *, method: str, model: Any, n_clusters: int,
                        extra: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze clustering results and compute metrics."""
        pass  # Implementation to be added
