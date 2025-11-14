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
    scaler: Any = field(default=None, repr=False)

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
        self.data = fetch_clustering_data(
            min_transactions=self.min_transactions,
            class3_description=self.class3_description,
            product_description=self.product_description,
        )
        self._data_loaded = True
        print(f"✅ Data loaded with {len(self.data)} records.")

    def prepare_features(self) -> Tuple[pd.DataFrame, List[str]]:
        if not self._data_loaded:
            raise ValueError("❌ Data must be loaded first. Call load_data().")

        print("\n🔧 Step 2: Preparing features...")
        self.features, self.feature_columns = prepare_clustering_features(self.data)
        self.X_scaled, self.scaler = scale_features(self.features, self.feature_columns)
        self._features_prepared = True

        print("✅ Features prepared and standardized")
        print(f"   Total features: {len(self.feature_columns)}")
        print(
            f"   Purchase pattern features: {len([f for f in self.feature_columns if f.startswith('pct_qty_')])}"
        )
        print(f"   Standardized matrix shape: {self.X_scaled.shape}")
        return self.features, self.feature_columns

    def optimize_clusters(self) -> Dict[str, Any]:
        if not self._features_prepared:
            raise ValueError("❌ Features must be prepared first. Call prepare_features().")

        print("\n🔍 Step 3: Optimizing cluster numbers...")
        effective_max_clusters = min(self.max_clusters, max(2, len(self.features) // 10))
        self.optimization_results = find_optimal_clusters(
            self.X_scaled,
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
            labels, model = perform_kmeans_clustering(self.X_scaled, n_clusters, self.random_state)
            method_specific = {"model": model, "n_clusters": n_clusters}
        elif method == "hierarchical":
            labels, linkage_matrix = perform_hierarchical_clustering(self.X_scaled, n_clusters)
            method_specific = {"linkage_matrix": linkage_matrix, "n_clusters": n_clusters}
        else:
            labels, model = perform_dbscan_clustering(self.X_scaled, eps, min_samples)
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

        df_clustered, summary = analyze_clusters(
            self.features, labels, self.feature_columns, cluster_method=method.title()
        )

        labels_arr = np.asarray(labels)
        silhouette_avg: Optional[float] = None
        if method == "dbscan":
            non_noise = labels_arr != -1
            if non_noise.sum() > 1 and len(set(labels_arr[non_noise])) > 1:
                silhouette_avg = silhouette_score(self.X_scaled[non_noise], labels_arr[non_noise])
        else:
            if len(set(labels_arr)) > 1:
                silhouette_avg = silhouette_score(self.X_scaled, labels_arr)

        calinski_score: Optional[float] = None
        if method != "dbscan" or (-1 not in labels_arr and len(set(labels_arr)) > 1):
            calinski_score = calinski_harabasz_score(self.X_scaled, labels_arr)

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
                plot_cluster_visualization(self.X_scaled, res["labels"], self.features, method_name=method.title())
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
                plot_dbscan_analysis(self.X_scaled, res["labels"], self.features, res.get("eps"), res.get("min_samples"))
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
