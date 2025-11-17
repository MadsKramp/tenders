"""
Scalable Clustering Pipeline for Purchase Patterns Analysis

Structured pipeline to run clustering on product purchase patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    silhouette_score,
)

import warnings

warnings.filterwarnings("ignore")

# Project utilities
from .clustering_utils import (
    fetch_clustering_data,
    prepare_clustering_features,
    scale_features,
    find_optimal_clusters,
    perform_kmeans_clustering,
    perform_hierarchical_clustering,
    find_optimal_dbscan_params,
    perform_dbscan_clustering,
    analyze_clusters,
    plot_cluster_optimization_metrics,
    plot_cluster_visualization,
    plot_dendrogram,
    plot_purchase_patterns_by_cluster,
    plot_cluster_quantity_heatmap,
    plot_dbscan_analysis,
    save_clustering_results,
    save_clustering_results_bq,
)
from .abc_segmentation import (
    compute_abc_tiers,
    compute_kmeans_tiers,
    compute_gmm_tiers,
    summarize_tiers,
)

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from . import clustering_params as params


@dataclass
class ClusteringPipeline:
    """
    Comprehensive pipeline for clustering analysis on product purchase patterns.
    """

    # Configuration
    min_transactions: int = params.MIN_TRANSACTIONS
    class3_description: str = params.CLASS3_DESCRIPTION
    product_description: str = params.PRODUCT_DESCRIPTION
    max_clusters: int = params.MAX_CLUSTERS
    min_clusters: int = params.MIN_CLUSTERS
    random_state: int = params.RANDOM_STATE

    # Data containers
    data: Optional[pd.DataFrame] = field(default=None, repr=False)
    features: Optional[pd.DataFrame] = field(default=None, repr=False)
    feature_columns: Optional[List[str]] = field(default=None, repr=False)
    X_scaled: Optional[np.ndarray] = field(default=None, repr=False)
    X_for_clustering: Optional[np.ndarray] = field(default=None, repr=False)  # Possibly PCA-reduced matrix
    scaler: Any = field(default=None, repr=False)
    pca_model: Any = field(default=None, repr=False)
    sampled_indices: Optional[np.ndarray] = field(default=None, repr=False)

    # Results containers
    optimization_results: Dict[str, Any] = field(default_factory=dict)
    clustering_results: Dict[str, Any] = field(default_factory=dict)
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    export_files: Dict[str, str] = field(default_factory=dict)

    # Pipeline state
    _data_loaded: bool = field(default=False, init=False, repr=False)
    _features_prepared: bool = field(default=False, init=False, repr=False)
    _optimization_completed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        print("🚀 Clustering Pipeline initialized")
        print(f"   Min transactions: {self.min_transactions}")
        print(f"   Class filter: {self.class3_description}")
        print(f"   Product filter: {self.product_description}")

    # Internals
    def _determine_optimal_clusters(self) -> int:
        print("🔍 Determining optimal number of clusters...")
        if not self._optimization_completed:
            self.optimization_results = find_optimal_clusters(
                self.X_scaled,
                max_clusters=self.max_clusters,
                min_clusters=self.min_clusters,
                random_state=self.random_state,
            )
            self._optimization_completed = True

        opt = self.optimization_results
        for key in ("optimal_silhouette", "optimal_calinski", "optimal_elbow"):
            if key in opt and opt[key] is not None:
                k = int(opt[key])
                print(f"✅ Optimal number of clusters determined ({key}): {k}")
                return k
        fallback_k = max(self.min_clusters, 2)
        print(f"⚠️  No optimal key found; falling back to K={fallback_k}")
        return fallback_k

    # Steps
    def load_data(self) -> None:
        print("📥 Loading data...")
        # Decide aggregation based on pattern feature requirement
        aggregate = not params.INCLUDE_PATTERN_FEATURES
        self.data = fetch_clustering_data(
            min_transactions=self.min_transactions,
            class3_description=self.class3_description,
            product_description=self.product_description,
            aggregate=aggregate,
        )
        if params.INCLUDE_PATTERN_FEATURES:
            from .clustering_utils import build_pattern_features
            print("🔨 Building purchase pattern percentage features (pct_qty_*) from raw data...")
            self.data = build_pattern_features(self.data)
            print(f"✅ Pattern features added. Columns now: {len(self.data.columns)} (including pct_qty_*)")
        self._data_loaded = True
        print(f"✅ Data loaded with {len(self.data)} records.")

    def prepare_features(self) -> Tuple[pd.DataFrame, List[str]]:
        if not self._data_loaded:
            raise ValueError("❌ Data must be loaded first. Call load_data().")

        print("\n🔧 Step 2: Preparing features...")
        self.features, self.feature_columns = prepare_clustering_features(self.data)

        # Optional column pruning based on correlation
        if params.ENABLE_COLUMN_PRUNING:
            numeric_df = self.features.select_dtypes(include=[np.number])
            if numeric_df.shape[1] > 2:
                corr = numeric_df.corr().abs()
                to_drop = set()
                essentials = {"total_transactions", "total_spend", "avg_unit_purchase_price"}  # Always keep
                # Upper triangle iteration
                for i, col1 in enumerate(corr.columns):
                    if col1 in to_drop:
                        continue
                    for j, col2 in enumerate(corr.columns[i+1:], start=i+1):
                        if col2 in to_drop:
                            continue
                        if corr.iloc[i, j] >= params.COLUMN_PRUNE_CORRELATION_THRESHOLD:
                            # Decide which column to drop: prefer keeping essentials and a canonical metric
                            if col2 in essentials and col1 not in essentials:
                                to_drop.add(col1)
                            elif col1 in essentials:
                                to_drop.add(col2)
                            else:
                                # Prefer dropping duplicate counters like purchase_count over total_transactions
                                if col2 == "purchase_count":
                                    to_drop.add(col2)
                                elif col1 == "purchase_count":
                                    to_drop.add(col1)
                                else:
                                    to_drop.add(col2)
                if to_drop:
                    print(f"⚖️  Column pruning: dropping highly correlated columns (>={params.COLUMN_PRUNE_CORRELATION_THRESHOLD}): {', '.join(sorted(to_drop))}")
                    self.features.drop(columns=list(to_drop), inplace=True, errors='ignore')
                    self.feature_columns = [c for c in self.feature_columns if c not in to_drop]

        self.X_scaled, self.scaler = scale_features(self.features, self.feature_columns)
        # Impute missing numeric values before PCA/clustering (NaNs can arise from division by zero or single-transaction stddev)
        numeric_cols = self.features.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            nan_counts = self.features[numeric_cols].isna().sum()
            total_nans = int(nan_counts.sum())
            if total_nans > 0:
                # Specific zero-imputation candidates
                zero_fill = [c for c in ['std_purchase_amount_eur','avg_unit_price','purchase_quantity','purchase_amount_eur'] if c in numeric_cols]
                for c in zero_fill:
                    self.features[c] = self.features[c].fillna(0)
                # Median fill for remaining numeric columns with NaNs
                remaining = [c for c in numeric_cols if self.features[c].isna().any()]
                for c in remaining:
                    med = self.features[c].median()
                    self.features[c] = self.features[c].fillna(med)
                print(f"🛠️ Imputed {total_nans} missing numeric values (zero-fill: {', '.join(zero_fill)}; median-fill: {', '.join(remaining)})")
                # Re-scale after imputation to avoid mismatch
                self.X_scaled, self.scaler = scale_features(self.features, self.feature_columns)

        # Final safeguard: if scaled matrix still contains NaNs, perform direct array-level cleanup
        if self.X_scaled is not None and np.isnan(self.X_scaled).any():
            # Identify problematic columns by mapping back to numeric feature names
            numeric_feature_names = self.features.select_dtypes(include=[np.number]).columns.tolist()
            nan_mask = np.isnan(self.X_scaled)
            col_nan_counts = nan_mask.sum(axis=0)
            problematic = [f"{numeric_feature_names[i]}={col_nan_counts[i]}" for i in range(len(col_nan_counts)) if col_nan_counts[i] > 0 and i < len(numeric_feature_names)]
            print(f"⚠️ Detected residual NaNs after imputation & scaling: {sum(col_nan_counts)} total across columns: {', '.join(problematic)}")
            # Replace residual NaNs with 0 (neutral for StandardScaler output) and warn
            self.X_scaled = np.nan_to_num(self.X_scaled, nan=0.0)
            print("✅ Replaced remaining NaNs in scaled feature matrix with 0.0 to allow PCA to proceed.")

        # --- Single-column clustering override ---
        if params.CLUSTER_BY_SINGLE_COLUMN:
            target_col = params.SINGLE_COLUMN_NAME
            numeric_feature_names = self.features.select_dtypes(include=[np.number]).columns.tolist()
            if target_col not in numeric_feature_names:
                raise ValueError(f"Single-column clustering enabled, but column '{target_col}' not found in numeric features: {numeric_feature_names}")
            # Find index in numeric feature ordering used during scaling
            target_index = numeric_feature_names.index(target_col)
            single_feature_vector = self.X_scaled[:, [target_index]]  # keep 2D shape (n_samples, 1)
            self.X_for_clustering = single_feature_vector
            print(f"🎯 Single-column clustering active: using only '{target_col}' ({single_feature_vector.shape[0]} samples). Skipping PCA.")
        else:
            self.X_for_clustering = None  # will be set later by PCA or kept as full scaled

        # Optional PCA dimensionality reduction (skip if single-column clustering active)
        if params.USE_PCA and not params.CLUSTER_BY_SINGLE_COLUMN:
            pca_initial = PCA(random_state=self.random_state)
            pca_initial.fit(self.X_scaled)
            cum_var = np.cumsum(pca_initial.explained_variance_ratio_)
            n_components = int(np.searchsorted(cum_var, params.PCA_VARIANCE_THRESHOLD) + 1)
            self.pca_model = PCA(n_components=n_components, random_state=self.random_state)
            self.X_for_clustering = self.pca_model.fit_transform(self.X_scaled)
            print(f"🧪 PCA applied: {n_components} components retain {cum_var[n_components-1]:.2%} variance")
        else:
            if self.X_for_clustering is None:
                self.X_for_clustering = self.X_scaled

        self._features_prepared = True

        print("✅ Features prepared and standardized")
        print(f"   Total features: {len(self.feature_columns)}")
        print(
            f"   Purchase pattern features: {len([f for f in self.feature_columns if f.startswith('pct_qty_')])}"
        )
        if params.USE_PCA:
            print(f"   PCA-reduced matrix shape: {self.X_for_clustering.shape}")
        else:
            print(f"   Standardized matrix shape: {self.X_for_clustering.shape}")
        return self.features, self.feature_columns

    def optimize_clusters(self) -> Dict[str, Any]:
        if not self._features_prepared:
            raise ValueError("❌ Features must be prepared first. Call prepare_features().")

        print("\n🔍 Step 3: Optimizing cluster numbers...")
        effective_max_clusters = min(self.max_clusters, max(2, len(self.features) // 10))

        X_optimize = self.X_for_clustering
        # Optional sampling for optimization only
        if params.ENABLE_SAMPLING and X_optimize.shape[0] > params.MAX_SAMPLE_SIZE:
            desired = min(params.MAX_SAMPLE_SIZE, int(X_optimize.shape[0] * params.SAMPLING_FRACTION))
            rng = np.random.default_rng(self.random_state)
            self.sampled_indices = rng.choice(X_optimize.shape[0], size=desired, replace=False)
            X_optimize = X_optimize[self.sampled_indices]
            print(f"🧪 Sampling enabled: using {desired} rows (fraction={params.SAMPLING_FRACTION}, cap={params.MAX_SAMPLE_SIZE}) for optimization")
        elif params.ENABLE_SAMPLING and X_optimize.shape[0] <= params.MAX_SAMPLE_SIZE and params.SAMPLING_FRACTION < 1.0:
            desired = int(X_optimize.shape[0] * params.SAMPLING_FRACTION)
            if desired >= 10 and desired < X_optimize.shape[0]:
                rng = np.random.default_rng(self.random_state)
                self.sampled_indices = rng.choice(X_optimize.shape[0], size=desired, replace=False)
                X_optimize = X_optimize[self.sampled_indices]
                print(f"🧪 Sampling enabled: using {desired}/{self.X_for_clustering.shape[0]} rows for optimization")

        self.optimization_results = find_optimal_clusters(
            X_optimize,
            max_clusters=effective_max_clusters,
            min_clusters=self.min_clusters,
            random_state=self.random_state,
        )
        plot_cluster_optimization_metrics(self.optimization_results)
        self._optimization_completed = True
        print("✅ Cluster optimization completed")
        print(f"   Optimal by Silhouette: {self.optimization_results.get('optimal_silhouette')}")
        print(f"   Optimal by Calinski-Harabasz: {self.optimization_results.get('optimal_calinski')}")
        print(f"   Optimal by Elbow: {self.optimization_results.get('optimal_elbow')}")
        return self.optimization_results

    def run_clustering_method(
        self,
        method: str,
        n_clusters: Optional[int] = None,
        eps: Optional[float] = None,
        min_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        method = method.lower()
        valid_methods = ["kmeans", "hierarchical", "dbscan"]
        if method not in valid_methods:
            raise ValueError(f"❌ Invalid method '{method}'. Must be one of: {valid_methods}")

        if method in ["kmeans", "hierarchical"] and not self._optimization_completed:
            raise ValueError("❌ Cluster optimization must be completed first. Call optimize_clusters().")
        if method == "dbscan" and not self._features_prepared:
            raise ValueError("❌ Features must be prepared first. Call prepare_features().")

        print(f"\n🎯 Step 4: Running {method} clustering...")

        if method != "dbscan" and n_clusters is None:
            n_clusters = self._determine_optimal_clusters()

        method_specific: Dict[str, Any] = {}
        if method == "kmeans":
            if params.USE_MINIBATCH_KMEANS:
                print("⚡ Using MiniBatchKMeans for clustering")
                model = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    random_state=self.random_state,
                    batch_size=params.MINIBATCH_BATCH_SIZE,
                    max_iter=params.MINIBATCH_MAX_ITER,
                    n_init=10,
                )
                labels = model.fit_predict(self.X_for_clustering)
            else:
                labels, model = perform_kmeans_clustering(self.X_for_clustering, n_clusters, self.random_state)
            method_specific = {"model": model, "n_clusters": n_clusters}
        elif method == "hierarchical":
            labels, linkage_matrix = perform_hierarchical_clustering(self.X_for_clustering, n_clusters)
            method_specific = {"linkage_matrix": linkage_matrix, "n_clusters": n_clusters}
        else:
            labels, model = perform_dbscan_clustering(self.X_for_clustering, eps, min_samples)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int((np.asarray(labels) == -1).sum())
            method_specific = {
                "model": model,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_ratio": n_noise / len(labels) if len(labels) else 0.0,
                "eps": getattr(model, "eps", eps),
                "min_samples": getattr(model, "min_samples", min_samples),
            }

        # Use original aggregated data for cluster analysis (contains core metrics even if features were pruned)
        df_clustered, summary = analyze_clusters(
            self.data if self.data is not None else self.features,
            labels,
            self.feature_columns,
            cluster_method=method.title(),
        )

        labels_arr = np.asarray(labels)
        silhouette_avg: Optional[float] = None
        if method == "dbscan":
            non_noise = labels_arr != -1
            if non_noise.sum() > 1 and len(set(labels_arr[non_noise])) > 1:
                silhouette_avg = silhouette_score(self.X_for_clustering[non_noise], labels_arr[non_noise])
        else:
            if len(set(labels_arr)) > 1:
                silhouette_avg = silhouette_score(self.X_for_clustering, labels_arr)

        calinski_score: Optional[float] = None
        if method != "dbscan" or (-1 not in labels_arr and len(set(labels_arr)) > 1):
            calinski_score = calinski_harabasz_score(self.X_for_clustering, labels_arr)

        results = {
            "labels": labels_arr,
            "df_clustered": df_clustered,
            "summary": summary,
            "silhouette_score": silhouette_avg,
        }
        if calinski_score is not None:
            results["calinski_score"] = calinski_score
        results.update(method_specific)

        self.clustering_results[method] = results

        if method == "dbscan":
            print(f"✅ {method.UPPER()} clustering completed")
            print(f"   Clusters: {n_clusters}, Noise points: {method_specific['n_noise']}")
        else:
            print(f"✅ {method.title()} clustering completed with {n_clusters} clusters")
        return results

    def run_kmeans_clustering(self, n_clusters: Optional[int] = None) -> Dict[str, Any]:
        return self.run_clustering_method("kmeans", n_clusters=n_clusters)

    def run_hierarchical_clustering(self, n_clusters: Optional[int] = None) -> Dict[str, Any]:
        return self.run_clustering_method("hierarchical", n_clusters=n_clusters)

    def run_dbscan_clustering(self, eps: Optional[float] = None, min_samples: Optional[int] = None) -> Dict[str, Any]:
        return self.run_clustering_method("dbscan", eps=eps, min_samples=min_samples)

    def compare_clustering_methods(self) -> Dict[str, Any]:
        if not self.clustering_results:
            raise ValueError("❌ No clustering results available. Run clustering methods first.")
        print("\n📊 Step 5: Comparing clustering methods...")
        comparison: Dict[str, Any] = {}
        for method_name, res in self.clustering_results.items():
            comparison[method_name] = {
                "n_clusters": res.get("n_clusters", 0),
                "silhouette_score": res.get("silhouette_score"),
                "calinski_score": res.get("calinski_score"),
                "n_products": len(res["df_clustered"]),
            }
            if method_name == "dbscan":
                comparison[method_name]["n_noise"] = res.get("n_noise", 0)
                comparison[method_name]["noise_ratio"] = res.get("noise_ratio", 0.0)

        method_names = list(self.clustering_results.keys())
        if len(method_names) >= 2:
            correlations: Dict[str, float] = {}
            for i, m1 in enumerate(method_names):
                for m2 in method_names[i + 1 :]:
                    l1 = self.clustering_results[m1]["labels"]
                    l2 = self.clustering_results[m2]["labels"]
                    if len(l1) == len(l2):
                        correlations[f"{m1}_vs_{m2}"] = adjusted_rand_score(l1, l2)
            if correlations:
                comparison["method_correlations"] = correlations

        self.analysis_results["method_comparison"] = comparison
        print("✅ Method comparison completed")
        print(f"   Methods compared: {list(comparison.keys())}")
        return comparison

    def generate_visualizations(self, methods: Optional[List[str]] = None, show_plots: bool = True) -> Dict[str, Any]:
        if not self.clustering_results:
            raise ValueError("❌ No clustering results available. Run clustering methods first.")
        print("\n📈 Step 6: Generating visualizations...")
        if methods is None:
            methods = list(self.clustering_results.keys())
        visualizations: Dict[str, Any] = {}
        if self.optimization_results:
            print("   📊 Plotting optimization metrics...")
            if show_plots:
                plot_cluster_optimization_metrics(self.optimization_results)
            visualizations["optimization_metrics"] = True
        for method in methods:
            if method not in self.clustering_results:
                print(f"   ⚠️  Skipping {method} - no results available")
                continue
            res = self.clustering_results[method]
            mviz: Dict[str, bool] = {}
            print(f"   📊 Generating {method} visualizations...")
            if show_plots:
                # Ensure latest visualization logic (single-feature fallback) is used
                try:
                    plot_cluster_visualization(self.X_for_clustering, res["labels"], self.features, method_name=method.title())
                except ValueError as e:
                    # Provide graceful degradation if unexpected PCA errors occur
                    print(f"   ⚠️ Visualization primary path failed: {e}. Attempting fallback 1D plotting.")
                    import matplotlib.pyplot as plt
                    import pandas as pd
                    import numpy as np
                    X_local = self.X_for_clustering
                    if X_local is not None and X_local.shape[1] == 1:
                        fig, ax = plt.subplots(figsize=(10,4))
                        ax.scatter(X_local[:,0], np.zeros_like(X_local[:,0]), c=res['labels'], cmap='viridis', alpha=0.7)
                        ax.set_title(f"{method.title()} Clusters (Single Feature Fallback)")
                        ax.set_xlabel("Standardized Value")
                        ax.set_yticks([])
                        plt.tight_layout(); plt.show()
                        mviz["cluster_visualization_fallback"] = True
                    else:
                        print("   ❌ Fallback could not be applied (unexpected shape).")
            mviz["cluster_visualization"] = True
            if show_plots:
                plot_purchase_patterns_by_cluster(res["df_clustered"], cluster_method=method)
            mviz["purchase_patterns"] = True
            if show_plots:
                plot_cluster_quantity_heatmap(res["df_clustered"], cluster_method=method)
            mviz["quantity_heatmap"] = True
            if method == "hierarchical" and show_plots and "linkage_matrix" in res:
                plot_dendrogram(res["linkage_matrix"], self.features)
                mviz["dendrogram"] = True
            if method == "dbscan" and show_plots:
                plot_dbscan_analysis(self.X_for_clustering, res["labels"], self.features, res.get("eps"), res.get("min_samples"))
                mviz["dbscan_analysis"] = True
            visualizations[method] = mviz
        self.analysis_results["visualizations"] = visualizations
        print("✅ Visualizations generated")
        print(f"   Methods visualized: {list(visualizations.keys())}")
        return visualizations

    def export_results(self, methods: Optional[List[str]] = None) -> Dict[str, str]:
        if not self.clustering_results:
            raise ValueError("❌ No clustering results available. Run clustering methods first.")
        print("\n💾 Step 7: Exporting results...")
        if methods is None:
            methods = list(self.clustering_results.keys())
        exported: Dict[str, str] = {}
        for method in methods:
            if method not in self.clustering_results:
                print(f"   ⚠️  Skipping {method} - no results available")
                continue
            print(f"   💾 Exporting {method} results...")
            res = self.clustering_results[method]
            detailed_file, summary_file = save_clustering_results(res["df_clustered"], res["summary"], rounding_filter=None)
            exported[f"{method}_detailed"] = detailed_file
            exported[f"{method}_summary"] = summary_file
        self.export_files = exported
        print("✅ Results exported")
        print(f"   Files created: {len(exported)}")
        return exported

    def run_complete_analysis(
        self,
        n_clusters_override: Optional[int] = None,
        include_dbscan: bool = False,
        show_visualizations: bool = True,
        export_results: bool = False,
    ) -> Dict[str, Any]:
        print("🚀 Starting full clustering analysis pipeline...")
        print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.load_data()
        self.prepare_features()
        self.optimize_clusters()
        if params.INCLUDE_KMEANS:
            self.run_kmeans_clustering(n_clusters_override)
        if params.INCLUDE_HIERARCHICAL:
            self.run_hierarchical_clustering(n_clusters_override)
        if include_dbscan and params.INCLUDE_DBSCAN:
            self.run_dbscan_clustering()
        self.compare_clustering_methods()
        if show_visualizations:
            self.generate_visualizations()
        if export_results:
            self.export_results()
        final = {
            "min_transactions": self.min_transactions,
            "class3_description": self.class3_description,
            "product_description": self.product_description,
            "data_shape": None if self.data is None else self.data.shape,
            "feature_count": 0 if not self.feature_columns else len(self.feature_columns),
            "optimization_results": self.optimization_results,
            "clustering_results": self.clustering_results,
            "analysis_results": self.analysis_results,
            "export_files": self.export_files,
            "completion_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        print("\n✅ Full analysis pipeline completed successfully!")
        print(f"   Methods run: {list(self.clustering_results.keys())}")
        print(f"   Files exported: {len(self.export_files)}")
        print(f"   Completion time: {final['completion_time']}")
        return final

    def save_results_to_bq(self, method: str) -> None:
        method = method.lower()
        if method not in self.clustering_results:
            raise ValueError(f"❌ No results for method '{method}'. Run the method first.")
        print(f"\n💾 Saving {method} clustering results to BigQuery...")
        df_res = self.clustering_results[method]["df_clustered"].copy()
        cluster_col_map = {
            "kmeans": "kmeans_cluster",
            "hierarchical": "hierarchical_cluster",
            "dbscan": "dbscan_cluster",
        }
        col = cluster_col_map[method]
        if col not in df_res.columns:
            raise KeyError(
                f"Expected column '{col}' not found in results; verify your 'analyze_clusters' output."
            )
        payload = df_res[["ProductNumber", col]].rename(columns={"ProductNumber": "item_number", col: "cluster_id"})
        save_clustering_results_bq(
            payload,
            method=method,
            class3=self.class3_description,
            key_word_description=self.product_description,
        )
        print(f"✅ {method} clustering results saved to BigQuery successfully.")
        
    def run_abc_segmentation(
        self,
        spend_col: str = 'total_spend',
        thresholds: tuple = (0.80, 0.95),
        include_models: bool = True,
        kmeans_k: int = 3,
        gmm_max_components: int = 3,
    ) -> dict:
        """Compute ABC tiers and optional model-driven tiers (k-means / GMM) on a per-product spend column.

        Args:
            spend_col: Column name representing per-product spend.
            thresholds: (A_cut, B_cut) cumulative share thresholds delimiting A/B/C.
            include_models: If True also compute k-means and GMM tiers for comparison.
            kmeans_k: Number of clusters for k-means (if enabled).
            gmm_max_components: Max components for GMM BIC-based selection.
        Returns:
            Dict with DataFrames and summaries keyed by segmentation type.
        """
        if self.data is None or not self._data_loaded:
            raise ValueError("Data not loaded. Call load_data() first.")
        base_df = self.data.copy()

        # Attempt derivation of spend_col if missing and requesting default 'total_spend'
        if spend_col not in base_df.columns:
            if spend_col == 'total_spend' and {'purchase_amount_eur','ProductNumber'} <= set(base_df.columns):
                print("ℹ️ Deriving 'total_spend' from purchase_amount_eur per ProductNumber (sum).")
                derived = base_df.groupby('ProductNumber')['purchase_amount_eur'].sum().rename('total_spend')
                base_df = base_df.merge(derived, on='ProductNumber', how='left')
            else:
                raise KeyError(f"Spend column '{spend_col}' not found and cannot be derived.")

        print("\n🔁 Running ABC / Pareto spend tier segmentation...")
        abc_df = compute_abc_tiers(base_df, spend_col=spend_col, thresholds=thresholds)
        abc_summary = summarize_tiers(abc_df, spend_col=spend_col, tier_col='abc_tier')
        result: Dict[str, Any] = {
            'abc_df': abc_df,
            'abc_summary': abc_summary,
            'thresholds': thresholds,
            'spend_col': spend_col,
        }
        if include_models:
            try:
                km_df = compute_kmeans_tiers(base_df, spend_col=spend_col, k=kmeans_k)
                km_summary = summarize_tiers(km_df, spend_col=spend_col, tier_col='kmeans_tier')
                result['kmeans_df'] = km_df
                result['kmeans_summary'] = km_summary
            except Exception as e:
                print(f"⚠️ K-means tiering skipped: {e}")
            try:
                gmm_df = compute_gmm_tiers(base_df, spend_col=spend_col, max_components=gmm_max_components)
                gmm_summary = summarize_tiers(gmm_df, spend_col=spend_col, tier_col='gmm_tier')
                result['gmm_df'] = gmm_df
                result['gmm_summary'] = gmm_summary
            except Exception as e:
                print(f"⚠️ GMM tiering skipped: {e}")
        self.analysis_results['abc_segmentation'] = result
        print("✅ ABC segmentation completed.")
        print("   Spend column:", spend_col)
        print("   Thresholds (A,B):", thresholds)
        print("   Tier distribution (ABC):")
        print(abc_summary[['abc_tier','count','spend','share_pct']])
        return result
