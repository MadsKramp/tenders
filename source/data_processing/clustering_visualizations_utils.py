# src/utils/clustering_visualizations_utils.py

from __future__ import annotations
from typing import Dict, Any, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# ----------------------------------------------------------------------
# 1) Optimization panels (Elbow / Silhouette / Calinski-Harabasz)
#    Expects the dict produced by your optimizer (cluster_range, inertias, etc.)
# ----------------------------------------------------------------------

def plot_optimization_panels(opt: Dict[str, Any],
                             figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """
    Plot Elbow (inertia), Silhouette, and Calinski–Harabasz diagnostics
    from an optimization run.

    Parameters
    ----------
    opt : dict
        Keys expected: 'cluster_range', 'inertias', 'silhouette_scores', 'calinski_scores',
        'optimal_elbow', 'optimal_silhouette', 'optimal_calinski'
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    cr = opt.get("cluster_range", [])
    inertias = opt.get("inertias", [])
    sil = opt.get("silhouette_scores", [])
    ch = opt.get("calinski_scores", [])

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Silhouette
    axes[0].plot(cr, sil, marker="o")
    axes[0].axvline(opt.get("optimal_silhouette", None), ls="--", color="r", label="Optimal")
    axes[0].set_title("Silhouette vs K")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Silhouette")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    # Calinski–Harabasz
    axes[1].plot(cr, ch, marker="o")
    axes[1].axvline(opt.get("optimal_calinski", None), ls="--", color="r", label="Optimal")
    axes[1].set_title("Calinski–Harabasz vs K")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("C–H score")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    # Elbow (inertia)
    axes[2].plot(cr, inertias, marker="o")
    axes[2].axvline(opt.get("optimal_elbow", None), ls="--", color="r", label="Elbow")
    axes[2].set_title("Elbow (Inertia) vs K")
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("Inertia")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 2) PCA scatter (model-agnostic)
# ----------------------------------------------------------------------

def pca_scatter(X_scaled: np.ndarray,
                labels: Sequence[int],
                title: str = "Clusters (PCA 2D)",
                annotate_centroids: bool = False,
                centroid_points: Optional[np.ndarray] = None,
                sample: Optional[int] = None,
                random_state: int = 42) -> plt.Figure:
    """
    2D PCA projection colored by cluster labels. Optionally overlays centroids.

    Parameters
    ----------
    X_scaled : np.ndarray
        Standardized feature matrix (n_samples, n_features)
    labels : Sequence[int]
        Cluster labels for each row of X_scaled (DBSCAN noise can be -1)
    title : str
        Plot title
    annotate_centroids : bool
        If True, annotate centroid points (requires centroid_points)
    centroid_points : np.ndarray | None
        Centroid coordinates in original feature space; will be PCA-transformed
        on the same PCA used for X_scaled (shape: [n_clusters, n_features])
    sample : int | None
        If set, randomly subsample to this many points for a lighter plot
    random_state : int
        Random seed for subsampling

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    rng = np.random.default_rng(random_state)
    X = X_scaled
    y = np.asarray(labels)

    if sample is not None and sample < X.shape[0]:
        idx = rng.choice(X.shape[0], size=sample, replace=False)
        X = X[idx]
        y = y[idx]

    pca = PCA(n_components=2, random_state=random_state)
    Xp = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(Xp[:, 0], Xp[:, 1], c=y, cmap="viridis", s=18, alpha=0.8)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Cluster")

    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.grid(True, alpha=0.3)

    # Optional centroids
    if annotate_centroids and centroid_points is not None and centroid_points.ndim == 2:
        Cp = pca.transform(centroid_points)
        ax.scatter(Cp[:, 0], Cp[:, 1], s=140, c="red", marker="X", label="Centroid")
        for i, (x, y_) in enumerate(Cp):
            ax.annotate(f"C{i}", (x, y_), textcoords="offset points", xytext=(6, 6))
        ax.legend(loc="best")

    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 3) Quick bar of cluster sizes
# ----------------------------------------------------------------------

def cluster_sizes_bar(labels: Sequence[int], title: str = "Cluster sizes") -> plt.Figure:
    """
    Bar chart of counts per cluster (noise grouped if -1).
    """
    s = pd.Series(labels, name="cluster")
    counts = s.value_counts().sort_index()
    counts.index = counts.index.map(lambda x: "Noise/Outliers" if x == -1 else int(x))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(counts)), counts.values)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=0)
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.set_xlabel("Cluster")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 4) Product × Year heatmap ordered by cluster
#    Expects df with columns: ProductNumber, year_authorization, purchase_amount_eur (or renamed)
# ----------------------------------------------------------------------

def product_year_heatmap(df: pd.DataFrame,
                         labels_by_product: pd.Series,
                         amount_col: str = "Purchase Amount EUR",
                         year_col: str = "year_authorization",
                         product_col: str = "ProductNumber",
                         figsize: Tuple[int, int] = (11, 8)) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Build a product × year heatmap of summed amounts, ordering rows by cluster.

    Parameters
    ----------
    df : DataFrame
        Must have product, year, and amount columns
    labels_by_product : pd.Series
        Index = ProductNumber, values = cluster labels for that product
    amount_col, year_col, product_col : str
        Column names
    figsize : tuple
        Figure size

    Returns
    -------
    fig : Figure
    mat : DataFrame
        Pivot matrix used for the heatmap
    """
    # Attach cluster to each product and order
    lbl = labels_by_product.rename("cluster")
    order = lbl.sort_values().index

    mat = (
        df.pivot_table(index=product_col, columns=year_col, values=amount_col,
                       aggfunc="sum", fill_value=0.0)
          .reindex(order)
    )

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(mat, ax=ax, cmap="viridis", cbar_kws={"label": "EUR"})
    ax.set_title("Purchase Amount Heatmap (products ordered by cluster)")
    ax.set_xlabel("Year")
    ax.set_ylabel("ProductNumber")
    fig.tight_layout()
    return fig, mat


# ----------------------------------------------------------------------
# 5) Purchase-pattern heatmap per cluster (e.g., pct_* features)
# ----------------------------------------------------------------------

def pattern_heatmap(df_clustered: pd.DataFrame,
                    cluster_col: str,
                    feature_prefixes: Iterable[str] = ("pct_qty_", "pct_purchase_value_"),
                    figsize: Tuple[int, int] = (12, 8)) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Heatmap of mean purchase-pattern features per cluster.
    Handles DBSCAN noise (-1) by labeling it "Noise/Outliers".

    Parameters
    ----------
    df_clustered : DataFrame
        Must include cluster_col and a set of columns starting with one of feature_prefixes
    cluster_col : str
        Column with cluster IDs (may contain -1 for noise)
    feature_prefixes : iterable of str
        Prefixes to detect pattern columns, e.g., "pct_qty_", "pct_purchase_value_"
    figsize : tuple
        Figure size

    Returns
    -------
    fig : Figure
    mean_df : DataFrame
        Mean percentages per cluster for the detected pattern columns
    """
    # Collect columns
    patt_cols = [c for c in df_clustered.columns
                 if any(c.startswith(p) for p in feature_prefixes)]
    if not patt_cols:
        raise ValueError("No pattern columns found with the given prefixes.")

    # Group and average
    mean_df = (
        df_clustered
        .assign(_cluster=df_clustered[cluster_col].map(lambda x: "Noise/Outliers" if x == -1 else int(x)))
        .groupby("_cluster")[patt_cols]
        .mean()
        .sort_index()
    )

    # Nicify column names a bit
    clean_cols = {c: c.replace("pct_qty_", "Qty ").replace("pct_purchase_value_", "Value ") for c in patt_cols}
    mean_df = mean_df.rename(columns=clean_cols)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(mean_df, annot=False, cmap="YlGnBu")
    ax.set_title("Average purchase-pattern percentages by cluster")
    ax.set_xlabel("Pattern feature")
    ax.set_ylabel("Cluster")
    fig.tight_layout()
    return fig, mean_df


# ----------------------------------------------------------------------
# 6) Compact per-cluster profile table (means/stds for chosen metrics)
# ----------------------------------------------------------------------

def cluster_profile_table(df: pd.DataFrame,
                          labels: Sequence[int],
                          metrics: Sequence[str],
                          id_col: Optional[str] = None,
                          method_name: str = "kmeans") -> pd.DataFrame:
    """
    Create a tidy per-cluster summary table (means/stds) for selected metrics.

    Parameters
    ----------
    df : DataFrame
        Original data aligned with labels (same order/length)
    labels : sequence of int
        Cluster labels (DBSCAN noise allowed)
    metrics : sequence of str
        Numeric columns in df to summarize
    id_col : str | None
        Optional ID column to include record counts
    method_name : str
        For naming the cluster column

    Returns
    -------
    summary : DataFrame
        One row per cluster (incl. Noise/Outliers if present)
    """
    tmp = df.copy()
    tmp[f"{method_name.lower()}_cluster"] = np.asarray(labels)

    grp = tmp.groupby(f"{method_name.lower()}_cluster")
    means = grp[metrics].mean().add_prefix("mean_")
    stds = grp[metrics].std(ddof=0).add_prefix("std_")
    counts = grp.size().rename("n")

    summary = pd.concat([counts, means, stds], axis=1).reset_index()
    summary[f"{method_name.lower()}_cluster"] = summary[f"{method_name.lower()}_cluster"].map(
        lambda x: "Noise/Outliers" if x == -1 else int(x)
    )
    if id_col and id_col in df.columns:
        # How many unique IDs per cluster
        uniq = grp[id_col].nunique().rename("unique_ids")
        summary = summary.merge(uniq.reset_index(), on=f"{method_name.lower()}_cluster", how="left")
    return summary
