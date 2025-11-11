"""
clustering_pipeline.py
----------------------

Scalable clustering pipeline on tender material (super_table) using TWO features
derived from yearly columns according to `clustering_params.feature_spec()`:
    - feat_purchase_amount_eur (strategy: sum|latest|mean on purchase_amount_eur_YYYY)
    - feat_quantity_sold       (strategy: sum|latest|mean on quantity_sold_YYYY)

Data source (EU): kramp-sharedmasterdata-prd.MadsH.super_table

Simplified filtering surface (see analysis_utils.fetch_super_table_filtered):
    class_2 / class_3 / brand_name / GroupVendorName (via purchase_data) / description regex keywords.
Legacy filters (class_4, country of origin, purchase-stop) are ignored here.

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

# Simplified filtered fetch + helpers
try:
    from source.data_processing.analysis_utils import (
        fetch_super_table_filtered,
        compute_totals,
    )
except ImportError:
    from .analysis_utils import fetch_super_table_filtered, compute_totals  # type: ignore

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
# Year column helpers (derive from params feature spec)
# =============================================================================

def _build_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create clustering feature columns per params.feature_spec().

    Strategies: sum | latest | mean.
    Missing columns are ignored gracefully.
    """
    spec = params.feature_spec()
    work = df.copy()
    for feat_name, cfg in spec.items():
        cols = [c for c in cfg["from"] if c in work.columns]
        if not cols:
            work[feat_name] = 0.0
            continue
        block = work[cols].apply(pd.to_numeric, errors="coerce")
        strat = cfg.get("strategy", "sum").lower()
        if strat == "latest":
            # last non-null in chronological order
            work[feat_name] = block.ffill(axis=1).iloc[:, -1].fillna(0)
        elif strat == "mean":
            work[feat_name] = block.mean(axis=1, skipna=True).fillna(0)
        else:  # sum default
            work[feat_name] = block.sum(axis=1, skipna=True).fillna(0)
    return work


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
    feature_cols: List[str] = field(default_factory=lambda: list(params.CLUSTER_FEATURES))
    X_scaled: np.ndarray | None = None
    scaler: StandardScaler | None = None
    optimization: Dict[str, Any] = field(default_factory=dict)
    clustering_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # --------------------------
    # Step 1: Load filtered data
    # --------------------------
    def load_data(self) -> pd.DataFrame:
        """Fetch filtered super_table slice using simplified filtered fetch.

        Uses params.filtered_fetch_kwargs() and then computes totals for convenience.
        """
        kwargs = params.filtered_fetch_kwargs()
        df = fetch_super_table_filtered(**kwargs)
        if df is None or df.empty:
            raise ValueError("No data returned from BigQuery with current simplified filters.")
        # Add total helper columns (not necessarily used for features but handy for summaries)
        df = compute_totals(df)
        self.data = df
        return df

    # -----------------------------------
    # Step 2: Prepare the two clustering features
    # -----------------------------------
    def prepare_features(self, *, min_transactions: Optional[int] = params.MIN_TRANSACTIONS) -> Tuple[pd.DataFrame, List[str]]:
        if self.data is None:
            raise RuntimeError("Call load_data() first.")
        df = self.data.copy()

        # Derive feature columns per spec
        df = _build_feature_columns(df)

        # Optional transaction threshold using orders_total if present
        if min_transactions and "orders_total" in df.columns:
            df = df.loc[df["orders_total"] >= int(min_transactions)].copy()

        # Drop rows where both feature values are zero (no signal)
        if len(self.feature_cols) == 2:
            a, b = self.feature_cols
            df = df.loc[(df[a] > 0) | (df[b] > 0)].copy()

        feats = df[self.feature_cols].astype(float).fillna(0)
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(feats.values)
        self.scaler = scaler
        self.features = df
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
        """Attach labels and compute summary metrics.

        Returns dict with:
          df_clustered, cluster_summary, silhouette_score (if applicable),
          calinski_harabasz_score (if applicable), n_clusters, and extras.
        """
        if self.features is None:
            raise RuntimeError("Features not prepared.")
        lab_col = f"{method.lower()}_cluster"
        dfc = self.features.copy()
        dfc[lab_col] = labels

        # Summary (use median for robustness)
        summary = (
            dfc.groupby(lab_col, dropna=False)
               .agg(
                   n_items=("item_number", "nunique"),
                   **{f"median_{c}": (c, "median") for c in self.feature_cols},
                   **{f"mean_{c}": (c, "mean") for c in self.feature_cols},
               )
               .reset_index()
               .sort_values("n_items", ascending=False)
        )

        # Quality metrics (skip if degenerate or noise-only)
        sil = None
        ch = None
        try:
            # For DBSCAN include only non-noise points for silhouette if >1 cluster
            if method.lower() == "dbscan" and (-1 in labels):
                mask = labels != -1
                uniq = set(labels[mask])
                if len(uniq) > 1:
                    sil = float(silhouette_score(self.X_scaled[mask], labels[mask]))
                    ch = float(calinski_harabasz_score(self.X_scaled[mask], labels[mask]))
            else:
                if n_clusters > 1:
                    sil = float(silhouette_score(self.X_scaled, labels))
                    ch = float(calinski_harabasz_score(self.X_scaled, labels))
        except Exception:
            sil = None
            ch = None

        result = dict(
            df_clustered=dfc,
            cluster_summary=summary,
            silhouette_score=sil,
            calinski_harabasz=ch,
            n_clusters=n_clusters,
            feature_columns=self.feature_cols,
            method=method,
            extras=extra or {},
            model_repr=repr(model),
        )
        return result

    # -------------------------------------------------
    # Convenience orchestration
    # -------------------------------------------------
    def run_complete_analysis(self, *, include_dbscan: bool | None = None, show_visualizations: bool | None = None) -> Dict[str, Dict[str, Any]]:
        """Full pipeline shortcut.

        Returns clustering_results dict after running selected methods.
        """
        self.load_data()
        self.prepare_features()
        self.optimize_clusters()

        include_dbscan = params.INCLUDE_DBSCAN if include_dbscan is None else include_dbscan
        show_visualizations = params.SHOW_VISUALIZATIONS if show_visualizations is None else show_visualizations

        if self.include_kmeans:
            self.run_kmeans()
        if self.include_hierarchical:
            self.run_hierarchical()
        if include_dbscan and self.include_dbscan:
            self.run_dbscan()
        # (Visualization integration left to external plotting utilities)
        return self.clustering_results
