"""
Clustering Utilities for Purchase Patterns Analysis

This module provides specialized clustering functions for analyzing products based on
their purchase amounts in EUR, specifically focusing on value structure distribution
and product characteristics as requested in the products clustering analysis.
"""
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, List, Optional

from source.db_connect.bigquery_connector import BigQueryConnector
from . import clustering_params as params
from collections import Counter

# Load environment variables
load_dotenv()
PROJECT_ID = os.getenv('PROJECT_ID')
DATASET_ID = os.getenv('DATASET_ID')
TABLE_ID = os.getenv('TABLE_ID')

PUrchase_data_table = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

def fetch_clustering_data(
    min_transactions: int = 0,
    rounding_filter: Optional[float] = None,
    class3_description: Optional[str] = None,
    product_description: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch raw clustering data from BigQuery applying optional filters.

    This signature matches what the object oriented pipeline expects.
    Filters are applied in pandas after retrieval to keep the query simple
    (adjust later if performance requires server‑side filtering).
    """
    print("Fetching clustering data from BigQuery...")
    bq_client = BigQueryConnector()
    query = f"SELECT * FROM `{PUrchase_data_table}`"
    df = bq_client.query(query)
    if df is None:
        raise RuntimeError("BigQuery query returned no results; check credentials and table access.")

    # Defensive: only apply filters if the columns exist
    if 'total_transactions' in df.columns and min_transactions:
        df = df[df['total_transactions'] >= min_transactions]
    if rounding_filter is not None and 'Rounding' in df.columns:
        df = df[df['Rounding'] >= rounding_filter]
    if class3_description and 'Class3Description' in df.columns:
        df = df[df['Class3Description'].str.contains(class3_description, case=False, na=False)]
    if product_description and 'ProductDescription' in df.columns:
        df = df[df['ProductDescription'].str.contains(product_description, case=False, na=False)]

    print(f"✅ Retrieved {len(df)} records after filter application")
    return df

# Get centralized clustering parameters with paramters
def get_clustering_params() -> Dict[str, Any]:
    """
    Retrieve clustering parameters from the centralized configuration.
    
    Returns:
        Dict[str, Any]: Dictionary of clustering parameters.
    """
    return {
        'min_transactions': params.MIN_TRANSACTIONS,
        'class3_description': params.CLASS3_DESCRIPTION,
        'product_description': params.PRODUCT_DESCRIPTION,
              'max_clusters': params.MAX_CLUSTERS,
        'min_clusters': params.MIN_CLUSTERS,
        'random_state': params.RANDOM_STATE,
        'include_kmeans': params.INCLUDE_KMEANS,
        'include_hierarchical': params.INCLUDE_HIERARCHICAL,
        'include_dbscan': params.INCLUDE_DBSCAN,
        'dbscan_eps': params.DBSCAN_EPS,
        'dbscan_min_samples': params.DBSCAN_MIN_SAMPLES,
        'show_visualizations': params.SHOW_VISUALIZATIONS,
        'export_results': params.EXPORT_RESULTS
    }

def prepare_clustering_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Return feature dataframe + list of feature columns.

    Strict mode: only include numeric purchase pattern percentage columns
    (pct_qty_*) plus core numeric descriptors and optional numeric dimensions.
    """
    print("Preparing clustering features...")
    # Use explicitly requested pattern columns
    requested_pattern_cols = [
        'crm_main_group_vendor',
        'purchase_amount_eur',
        'purchase_quantity',
        'authorization_year',
    ]
    pattern_cols = [c for c in requested_pattern_cols if c in df.columns]
    base_cols = [
        c for c in [
            'avg_purchase_amount_eur', 'std_purchase_amount_eur', 'purchase_count',
            'total_transactions', 'total_turnover', 'avg_unit_price'
        ] if c in df.columns
    ]
    dimension_cols = [
        c for c in ['length_mm', 'diameter_mm', 'thread_pitch_mm', 'length', 'diameter', 'thread_pitch']
        if c in df.columns
    ]
    feature_cols = pattern_cols + base_cols + dimension_cols
    if not feature_cols:
        raise ValueError("No usable feature columns found for clustering.")
    features = df[feature_cols].copy()
    return features, feature_cols

def scale_features(features: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, StandardScaler]:
    """
    Scale features using StandardScaler.
    
    Args:
        features (pd.DataFrame): Feature DataFrame.
        
    Returns:
        Tuple[np.ndarray, StandardScaler]: Scaled feature array and fitted scaler.
    """
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    return X_scaled, scaler

def find_optimal_clusters(X: np.ndarray, max_clusters: int = 10, min_clusters: int = 2, random_state: int = 42) -> Dict[str, Any]:
    """
    Find the optimal number of clusters using silhouette and Calinski-Harabasz scores.
    
    Args:
        X (np.ndarray): Scaled feature array.
        min_clusters (int): Minimum number of clusters to test.
        max_clusters (int): Maximum number of clusters to test.
        random_state (int): Random state for reproducibility.
        
    Returns:
        Dict[str, List]: Dictionary containing cluster numbers and corresponding scores.
    """
    print("🔍 Finding optimal number of clusters...")
    
    silhouette_scores = []
    calinski_scores = []
    inertias = []
    cluster_range = range(min_clusters, min(max_clusters + 1, len(X) // 10))  
    
    # Ensure reasonable cluster sizes
    
    for n_clusters in cluster_range:
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Calculate metrics
        silhouette_avg = silhouette_score(X, cluster_labels)
        calinski_score = calinski_harabasz_score(X, cluster_labels)
        
        silhouette_scores.append(silhouette_avg)
        calinski_scores.append(calinski_score)
        inertias.append(kmeans.inertia_)
        
        print(f"   {n_clusters} clusters: Silhouette={silhouette_avg:.3f}, Calinski-Harabasz={calinski_score:.1f}")
    
    # Find optimal based on silhouette score
    optimal_silhouette_idx = np.argmax(silhouette_scores)
    optimal_silhouette = cluster_range[optimal_silhouette_idx]
    
    # Find optimal based on Calinski-Harabasz score
    optimal_calinski_idx = np.argmax(calinski_scores)
    optimal_calinski = cluster_range[optimal_calinski_idx]
    
    # Elbow method for inertia (simple heuristic)
    if len(inertias) >= 3:
        # Find the point where improvement slows down significantly
        improvements = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
        improvement_ratios = [improvements[i] / improvements[i+1] if improvements[i+1] > 0 else 1 for i in range(len(improvements)-1)]
        if improvement_ratios:
            elbow_idx = np.argmax(improvement_ratios)
            optimal_elbow = cluster_range[elbow_idx + 1]
        else:
            optimal_elbow = cluster_range[len(cluster_range)//2]
    else:
        optimal_elbow = cluster_range[0]
    
    results = {
        'optimal_silhouette': optimal_silhouette,
        'optimal_calinski': optimal_calinski,
        'optimal_elbow': optimal_elbow,
        'silhouette_scores': silhouette_scores,
        'calinski_scores': calinski_scores,
        'inertias': inertias,
        'cluster_range': list(cluster_range)
    }
    
    print(f"✅ Optimal clusters - Silhouette: {optimal_silhouette}, Calinski-Harabasz: {optimal_calinski}, Elbow: {optimal_elbow}")
    
    return results

def perform_kmeans_clustering(X: np.ndarray, n_clusters: int, random_state: int=42) -> Tuple[np.ndarray, Any]:
    """
    Perform K-means clustering.
    
    Args:
        X: Standardized feature matrix
        n_clusters: Number of clusters
        
    Returns:
        Tuple of (cluster labels, fitted model)
    """
    print(f"🎯 Performing K-means clustering with {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Calculate metrics
    silhouette_avg = silhouette_score(X, cluster_labels)
    calinski_score = calinski_harabasz_score(X, cluster_labels)
    
    print(f"✅ K-means completed - Silhouette: {silhouette_avg:.3f}, Calinski-Harabasz: {calinski_score:.1f}")
    
    return cluster_labels, kmeans


def perform_hierarchical_clustering(X: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, Any]:
    """
    Perform hierarchical clustering.
    
    Args:
        X: Standardized feature matrix
        n_clusters: Number of clusters
        
    Returns:
        Tuple of (cluster labels, linkage matrix)
    """
    print(f"🌳 Performing hierarchical clustering with {n_clusters} clusters...")
    
    # Calculate linkage matrix
    linkage_matrix = linkage(X, method='ward')
    
    # Perform clustering
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = hierarchical.fit_predict(X)
    
    # Calculate metrics
    silhouette_avg = silhouette_score(X, cluster_labels)
    calinski_score = calinski_harabasz_score(X, cluster_labels)
    
    print(f"✅ Hierarchical clustering completed - Silhouette: {silhouette_avg:.3f}, Calinski-Harabasz: {calinski_score:.1f}")
    
    return cluster_labels, linkage_matrix


def find_optimal_dbscan_params(X: np.ndarray, min_samples_range: List[int] = None, 
                              eps_range: List[float] = None) -> Dict[str, Any]:
    """
    Find optimal parameters for DBSCAN clustering using k-distance graph and silhouette analysis.
    
    Args:
        X: Standardized feature matrix
        min_samples_range: Range of min_samples values to test
        eps_range: Range of eps values to test
        
    Returns:
        Dictionary with optimal parameters and evaluation metrics
    """
    print("🔍 Finding optimal DBSCAN parameters...")
    
    if min_samples_range is None:
        # Rule of thumb: min_samples = 2 * n_features
        min_samples_range = [max(2, 2 * X.shape[1]), max(3, 3 * X.shape[1]), 
                            max(5, int(0.1 * X.shape[0])), max(10, int(0.05 * X.shape[0]))]
    
    # Use k-distance graph to estimate eps if not provided
    if eps_range is None:
        k = max(min_samples_range)
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(X)
        distances, indices = neighbors_fit.kneighbors(X)
        distances = np.sort(distances[:, k-1], axis=0)
        
        # Find knee point in k-distance graph (simplified approach)
        knee_idx = np.argmax(np.diff(distances))
        optimal_eps = distances[knee_idx]
        print(f"   Estimated optimal eps from k-distance graph: {optimal_eps}")
        
        # Create range around the knee point
        eps_range = [optimal_eps * 0.5, optimal_eps * 0.75, optimal_eps, 
                    optimal_eps * 1.25, optimal_eps * 1.5]
        
        print(f"   Estimated eps range from k-distance graph: {eps_range}")
    
    best_params = {}
    best_score = -1
    results = []
    
    print(f"   Testing {len(min_samples_range)} min_samples × {len(eps_range)} eps combinations...")
    
    for min_samples in min_samples_range:
        for eps in eps_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            # Skip if no clusters found or all points are noise
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            if n_clusters < 2 or n_clusters > len(X) // 2:
                continue
                
            # Calculate silhouette score (excluding noise points)
            if len(set(labels)) > 1 and -1 not in labels:
                silhouette_avg = silhouette_score(X, labels)
            elif len(set(labels)) > 1:
                # Calculate silhouette for non-noise points only
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) > 1 and len(set(labels[non_noise_mask])) > 1:
                    silhouette_avg = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
                else:
                    silhouette_avg = -1
            else:
                silhouette_avg = -1
            
            # Custom score: balance silhouette score with reasonable noise ratio
            noise_ratio = n_noise / len(X)
            adjusted_score = silhouette_avg * (1 - min(noise_ratio, 0.5))  # Penalize high noise
            
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_ratio': noise_ratio,
                'silhouette_score': silhouette_avg,
                'adjusted_score': adjusted_score
            })
            
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_params = {
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'silhouette_score': silhouette_avg,
                    'adjusted_score': adjusted_score
                }
    
    results_df = pd.DataFrame(results)
    
    print(f"✅ DBSCAN parameter optimization completed")
    print(f"   Best parameters: eps={best_params.get('eps', 'N/A')}, min_samples={best_params.get('min_samples', 'N/A')}")
    print(f"   Best silhouette score: {best_params.get('silhouette_score', -1)}")
    print(f"   Expected clusters: {best_params.get('n_clusters', 0)}")
    
    return {
        'optimal_params': best_params,
        'all_results': results_df,
        'eps_range': eps_range,
        'min_samples_range': min_samples_range
    }


def perform_dbscan_clustering(X: np.ndarray, eps: float = None, min_samples: int = None) -> Tuple[np.ndarray, Any]:
    """
    Perform DBSCAN clustering.
    
    Args:
        X: Standardized feature matrix
        eps: Maximum distance between two samples for one to be considered as in the neighborhood of the other
        min_samples: Number of samples in a neighborhood for a point to be considered as a core point
        
    Returns:
        Tuple of (cluster labels, dbscan model)
    """
    # Auto-determine parameters if not provided
    if eps is None or min_samples is None:
        print("🔍 Auto-determining DBSCAN parameters...")
        optimization_results = find_optimal_dbscan_params(X)
        optimal_params = optimization_results['optimal_params']
        
        if not optimal_params:
            print("⚠️  Could not find optimal parameters, using defaults")
            eps = eps or 0.5
            min_samples = min_samples or max(2, 2 * X.shape[1])
        else:
            eps = eps or optimal_params['eps']
            min_samples = min_samples or optimal_params['min_samples']
    
    print(f"🎯 Performing DBSCAN clustering with eps={eps:.3f}, min_samples={min_samples}...")
    
    # Perform clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X)
    
    # Calculate metrics
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    noise_ratio = n_noise / len(X)
    
    print(f"✅ DBSCAN clustering completed:")
    print(f"   Clusters found: {n_clusters}")
    print(f"   Noise points: {n_noise} ({noise_ratio:.1%})")
    
    if n_clusters > 1 and -1 not in cluster_labels:
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(f"   Silhouette score: {silhouette_avg:.3f}")
    elif n_clusters > 1:
        # Calculate silhouette for non-noise points only
        non_noise_mask = cluster_labels != -1
        if np.sum(non_noise_mask) > 1 and len(set(cluster_labels[non_noise_mask])) > 1:
            silhouette_avg = silhouette_score(X[non_noise_mask], cluster_labels[non_noise_mask])
            print(f"   Silhouette score (non-noise): {silhouette_avg:.3f}")
    
    return cluster_labels, dbscan


def analyze_clusters(df: pd.DataFrame, cluster_labels: np.ndarray, feature_columns: List[str], 
                    cluster_method: str = "Kmeans") -> pd.DataFrame:
    """
    Analyze cluster characteristics and create summary statistics.
    
    Args:
        df: Original DataFrame with product data
        cluster_labels: Cluster assignments
        feature_columns: List of feature column names
        cluster_method: Name of clustering method used
        
    Returns:
        DataFrame with cluster analysis results
    """
    print(f"📊 Analyzing {cluster_method} cluster characteristics...")
    
    # Add cluster labels to dataframe
    df_clustered = df.copy()
    df_clustered[f'{cluster_method.lower()}_cluster'] = cluster_labels
    
    # Cluster summary statistics
    cluster_summary = []
    
    # Handle DBSCAN noise points specially
    unique_clusters = sorted(df_clustered[f'{cluster_method.lower()}_cluster'].unique())
    
    for cluster_id in unique_clusters:
        cluster_data = df_clustered[df_clustered[f'{cluster_method.lower()}_cluster'] == cluster_id]
        
        # Special handling for DBSCAN noise points (cluster_id = -1)
        cluster_label = "Noise/Outliers" if cluster_id == -1 else f"Cluster {cluster_id}"
        
        summary = {
            'cluster_id': cluster_id,
            'cluster_label': cluster_label,
            'cluster_method': cluster_method,
            'n_products': len(cluster_data),
            'avg_total_transactions': cluster_data['total_transactions'].mean(),
            'avg_total_turnover': cluster_data['total_turnover'].mean(),
            'avg_unit_price': cluster_data['avg_unit_price'].mean(),
            
        }
        
        # Add top purchase patterns for this cluster
        purchase_pattern_cols = [col for col in feature_columns if col.startswith('pct_qty_')]
        for col in purchase_pattern_cols:
            summary[f'avg_{col}'] = cluster_data[col].mean()
        
        
        
        cluster_summary.append(summary)
    
    cluster_summary_df = pd.DataFrame(cluster_summary)
    
    print(f"✅ Cluster analysis completed for {len(cluster_summary_df)} clusters")
    
    return df_clustered, cluster_summary_df


def plot_cluster_optimization_metrics(optimization_results: Dict[str, Any], figsize: Tuple[int, int] = (15, 5)):
    """
    Plot metrics for determining optimal number of clusters.
    
    Args:
        optimization_results: Results from find_optimal_clusters function
        figsize: Figure size for the plots
    """
    cluster_range = optimization_results['cluster_range']
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Silhouette Score
    axes[0].plot(cluster_range, optimization_results['silhouette_scores'], 'bo-')
    axes[0].set_title('Silhouette Score vs Number of Clusters')
    axes[0].set_xlabel('Number of Clusters')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].axvline(x=optimization_results['optimal_silhouette'], color='red', linestyle='--', 
                   label=f"Optimal: {optimization_results['optimal_silhouette']}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Calinski-Harabasz Score
    axes[1].plot(cluster_range, optimization_results['calinski_scores'], 'go-')
    axes[1].set_title('Calinski-Harabasz Score vs Number of Clusters')
    axes[1].set_xlabel('Number of Clusters')
    axes[1].set_ylabel('Calinski-Harabasz Score')
    axes[1].axvline(x=optimization_results['optimal_calinski'], color='red', linestyle='--',
                   label=f"Optimal: {optimization_results['optimal_calinski']}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Elbow Method (Inertia)
    axes[2].plot(cluster_range, optimization_results['inertias'], 'ro-')
    axes[2].set_title('Elbow Method (Inertia) vs Number of Clusters')
    axes[2].set_xlabel('Number of Clusters')
    axes[2].set_ylabel('Inertia')
    axes[2].axvline(x=optimization_results['optimal_elbow'], color='red', linestyle='--',
                   label=f"Optimal: {optimization_results['optimal_elbow']}")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_cluster_visualization(X_scaled: np.ndarray, cluster_labels: np.ndarray, df: pd.DataFrame,
                             method_name: str = "K-means", figsize: Tuple[int, int] = (15, 10)):
    """
    Visualize clusters using PCA and various scatter plots.
    
    Args:
        X_scaled: Standardized feature matrix
        cluster_labels: Cluster assignments
        df: Original DataFrame
        method_name: Name of clustering method
        figsize: Figure size for the plots
    """
    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculate explained variance
    explained_var = pca.explained_variance_ratio_
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # PCA scatter plot
    scatter = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    axes[0, 0].set_title(f'{method_name} Clusters (PCA Projection)')
    axes[0, 0].set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
    axes[0, 0].set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
    plt.colorbar(scatter, ax=axes[0, 0])
    
    # Cluster vs Total Transactions
    axes[0, 1].scatter(df['total_transactions'], df['avg_unit_price'], c=cluster_labels, cmap='viridis', alpha=0.7)
    axes[0, 1].set_title(f'{method_name} Clusters: Transactions vs Unit Price')
    axes[0, 1].set_xlabel('Total Transactions')
    axes[0, 1].set_ylabel('Average Unit Price (EUR)')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    
    # Cluster vs Rounding
    axes[1, 0].scatter(df['Rounding'], df['total_turnover'], c=cluster_labels, cmap='viridis', alpha=0.7)
    axes[1, 0].set_title(f'{method_name} Clusters: Rounding vs Total Turnover')
    axes[1, 0].set_xlabel('Rounding')
    axes[1, 0].set_ylabel('Total Turnover (EUR)')
    axes[1, 0].set_yscale('log')
    
    # Cluster size distribution
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    axes[1, 1].bar(cluster_counts.index, cluster_counts.values, color=plt.cm.viridis(cluster_counts.index / cluster_counts.index.max()))
    axes[1, 1].set_title(f'{method_name} Cluster Size Distribution')
    axes[1, 1].set_xlabel('Cluster ID')
    axes[1, 1].set_ylabel('Number of Products')
    
    plt.tight_layout()
    plt.show()


def plot_dendrogram(linkage_matrix: np.ndarray, df: pd.DataFrame, max_display: int = 30):
    """
    Plot dendrogram for hierarchical clustering.
    
    Args:
        linkage_matrix: Linkage matrix from hierarchical clustering
        df: Original DataFrame
        max_display: Maximum number of leaf nodes to display
    """
    plt.figure(figsize=(15, 8))
    
    # Create dendrogram
    dendrogram(linkage_matrix, 
              truncate_mode='level', 
              p=max_display,
              leaf_rotation=90,
              leaf_font_size=10)
    
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index or Cluster Size')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()

def plot_purchase_patterns_by_cluster(
    df_clustered: pd.DataFrame,
    cluster_method: str = "kmeans",
    figsize: Tuple[int, int] = (15, 10),
):
    """Wrapper matching pipeline call style.

    Automatically discovers purchase pattern columns (pct_qty_*) and
    plots average percentages per cluster.
    """
    cluster_column = f"{cluster_method.lower()}_cluster"
    if cluster_column not in df_clustered.columns:
        print(f"⚠️ Cluster column '{cluster_column}' not found; skipping purchase pattern plot.")
        return
    feature_columns = [c for c in df_clustered.columns if c.startswith("pct_qty_")]
    if not feature_columns:
        print("⚠️ No purchase pattern (pct_qty_*) columns found; skipping plot.")
        return
    clusters = sorted(df_clustered[cluster_column].unique())
    n_clusters = len(clusters)
    fig, axes = plt.subplots(n_clusters, 1, figsize=figsize, sharex=True)
    if n_clusters == 1:
        axes = [axes]
    for i, cluster_id in enumerate(clusters):
        cluster_data = df_clustered[df_clustered[cluster_column] == cluster_id]
        avg_patterns = cluster_data[feature_columns].mean()
        axes[i].bar(feature_columns, avg_patterns, color=plt.cm.viridis(i / max(1, n_clusters - 1)))
        axes[i].set_title(f"Average Purchase Patterns - {cluster_method.title()} Cluster {cluster_id}")
        axes[i].set_ylabel("Avg %")
        axes[i].set_ylim(0, 100)
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Purchase Quantity Categories")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_cluster_value_heatmap(df_clustered: pd.DataFrame, cluster_method: str = "kmeans"):
    """
    Plot heatmap showing distribution of transactions per purchase value size mapping for each cluster.
    
    Args:
        df_clustered: DataFrame with cluster assignments and purchase pattern percentages
        cluster_method: Name of clustering method (for column naming)
    """
    cluster_col = f'{cluster_method.lower()}_cluster'
    purchase_pattern_cols = [col for col in df_clustered.columns if col.startswith('pct_qty_')]
    
    heatmap_data = df_clustered.groupby(cluster_col)[purchase_pattern_cols].mean()
    if heatmap_data.empty:
        print("⚠️ No data for heatmap.")
        return
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu")
    plt.title(f"Purchase Pattern Heatmap by {cluster_method.title()} Clusters")
    plt.xlabel("Purchase Quantity Categories")
    plt.ylabel("Cluster ID")
    plt.tight_layout()
    plt.show()

def plot_dbscan_analysis(X: np.ndarray, cluster_labels: np.ndarray, df: pd.DataFrame, 
                        eps: float, min_samples: int, figsize: Tuple[int, int] = (15, 10)):
    """
    Create specialized visualizations for DBSCAN clustering results.
    
    Args:
        X: Standardized feature matrix used for clustering
        cluster_labels: DBSCAN cluster assignments
        df: Original DataFrame with product data
        eps: DBSCAN eps parameter used
        min_samples: DBSCAN min_samples parameter used
        figsize: Figure size for the plots
    """
    from sklearn.decomposition import PCA
    
    # Calculate DBSCAN metrics
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    noise_ratio = n_noise / len(cluster_labels)
    
    print(f"🎯 DBSCAN Analysis Visualization")
    print(f"   Parameters: eps={eps:.3f}, min_samples={min_samples}")
    print(f"   Clusters: {n_clusters}, Noise points: {n_noise} ({noise_ratio:.1%})")
    
    # Create PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # 1. Cluster visualization with noise points highlighted
    unique_labels = set(cluster_labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Noise points in black
            class_member_mask = (cluster_labels == k)
            xy = X_pca[class_member_mask]
            axes[0, 0].scatter(xy[:, 0], xy[:, 1], c='black', marker='x', s=20, alpha=0.8, label='Noise')
        else:
            class_member_mask = (cluster_labels == k)
            xy = X_pca[class_member_mask]
            axes[0, 0].scatter(xy[:, 0], xy[:, 1], c=[col], s=30, alpha=0.8, label=f'Cluster {k}')
    
    axes[0, 0].set_title(f'DBSCAN Clustering Results\n{n_clusters} clusters, {n_noise} noise points')
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f} variance)')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f} variance)')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    # 2. Purchase value distribution by cluster
    purchase_pattern_cols = [col for col in df.columns if col.startswith('pct_purchase_value_')]
    for cluster_id in sorted(unique_labels):
        cluster_data = df[cluster_labels == cluster_id]
        if cluster_id == -1:
            label = 'Noise'
        else:
            label = f'Cluster {cluster_id}'
        avg_patterns = cluster_data[purchase_pattern_cols].mean()
        axes[0, 1].plot(purchase_pattern_cols, avg_patterns, marker='o', label=label)
    axes[0, 1].set_title('Average Purchase Value Distribution by Cluster')
    axes[0, 1].set_xlabel('Purchase Value Categories')
    axes[0, 1].set_ylabel('Average Percentage')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Purchase price distribution by cluster
    for cluster_id in sorted(unique_labels):
        cluster_data = df[cluster_labels == cluster_id]
        if cluster_id == -1:
            label = 'Noise'
        else:
            label = f'Cluster {cluster_id}'
        axes[0, 2].hist(cluster_data['avg_unit_price'], bins=30, alpha=0.5, label=label)
    axes[0, 2].set_title('Average Unit Price Distribution by Cluster')
    axes[0, 2].set_xlabel('Average Unit Price')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 2].grid(True, alpha=0.3)

def plot_cluster_quantity_heatmap(df_clustered: pd.DataFrame, cluster_method: str = 'kmeans'):
    """Alias wrapper expected by pipeline (calls value heatmap)."""
    plot_cluster_value_heatmap(df_clustered, cluster_method=cluster_method)

def plot_sales_patterns_by_cluster(*args, **kwargs):
    """Backward compatibility alias for older name. Redirects to purchase pattern plot."""
    return plot_purchase_patterns_by_cluster(*args, **kwargs)
    
def save_clustering_results_bq(
    clustering_results: pd.DataFrame,
    method: str = 'kmeans',
    class3: Optional[str] = None,
    key_word_description: Optional[str] = None,
) -> None:
    """
    Save clustering results to BigQuery using BigQueryConnector.

    Expects a DataFrame with at least columns `item_number` and `cluster_id`.
    Adds optional context columns and writes into dataset `product_clustering`
    and table `{method}_clusters`.
    """
    print(f"💾 Saving {method} clustering results to BigQuery...")
    results_df = clustering_results.copy()
    # Optional context
    if class3:
        results_df['Class3'] = class3
    if key_word_description:
        results_df['KeyWordDescription'] = key_word_description

    # Validate required columns
    required = {'item_number', 'cluster_id'}
    if not required.issubset(set(results_df.columns)):
        missing = required - set(results_df.columns)
        raise KeyError(f"Missing required columns for BQ save: {missing}")

    # Persist to BigQuery
    bq = BigQueryConnector()
    dataset_id = 'product_clustering'
    table_id = f"{method.lower()}_clusters"
    # Use append to accumulate runs; change to WRITE_TRUNCATE if needed
    ok = bq.save_to_table(results_df, dataset_id=dataset_id, table_id=table_id, write_disposition='WRITE_APPEND')
    if ok:
        print(f"✓ Saved clustering results to {bq.project_id}.{dataset_id}.{table_id}")
    else:
        print(f"✗ Failed to save clustering results to BigQuery")

def prepare_clustering_dataframe(clustering_results: pd.DataFrame, method: str = 'kmeans', random_state: int = 42
                                 , class3: Optional[str] = None, key_word_description: Optional[str] = None
                                 ) -> pd.DataFrame:
    """
    Process clustering results DataFrame for further analysis or saving.
    """
    print(f"🔧 Preparing {method} clustering results DataFrame...")
    results_df = clustering_results.copy()
    results_df.rename(columns={
        f'{method.lower()}_cluster': 'cluster_id',
        'product_number': 'ProductNumber',
        'product_description': 'ProductDescription'
    }, inplace=True)

def save_clustering_results(df_clustered: pd.DataFrame, cluster_summary: pd.DataFrame, 

                          rounding_filter: Optional[float] = None):
    """
    Save clustering results to Excel files with timestamp.
    
    Args:
        df_clustered: DataFrame with cluster assignments
        cluster_summary: Cluster summary statistics
       
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rounding_suffix = f"_rounding_{rounding_filter}" if rounding_filter is not None else ""
    clustered_filename = f'clustering_results{rounding_suffix}_{timestamp}.xlsx'
    summary_filename = f'cluster_summary{rounding_suffix}_{timestamp}.xlsx'
    print(f"💾 Saving clustering results to {clustered_filename} and {summary_filename}...")
    df_clustered.to_excel(clustered_filename, index=False)
    cluster_summary.to_excel(summary_filename, index=False)
    return clustered_filename, summary_filename
