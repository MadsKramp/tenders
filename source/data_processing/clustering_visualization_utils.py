"""
clustering_visualizations_utils.py

Visual tools & summaries for 2D tender-material clustering.

Designed for the Tender Material Generator stack:
- Clusters are built on two features derived from year-suffixed columns
  (purchase_amount_eur_YYYY, quantity_sold_YYYY) as in your pipeline.
- Figures use pure Matplotlib (no seaborn dependency).
- Each function prints concise business insights and returns tidy DataFrames/
  dicts so notebooks can reuse the numbers.

Key functions
-------------
- build_totals(df): derive totals/means from year-suffixed columns (idempotent)
- plot_overall_cluster_metrics(df): size, total purchase amount, total qty
- plot_yearly_trends(df): stacked/line trends per cluster by year
- plot_feature_scatter(df): feature scatter with cluster colors & centroids
- plot_origin_heatmap(df): STEP_CountryOfOrigin × cluster heatmap (ratio)
- plot_brand_supplier_bars(df): top brands & suppliers by cluster
- summarize_cluster_tables(df): compact per-cluster tables (top-N helpers)

Inputs
------
- df: DataFrame with at least:
    item_number, kmeans_cluster (or 'cluster'), and year-suffixed columns:
    purchase_amount_eur_2020..2024, quantity_sold_2021..2024 (as available)
  Optional but used when present:
    avg_price_eur_YYYY, brand_name, supplier_name_string, STEP_CountryOfOrigin

Notes
-----
- Works with ClusteringPipeline.clustering_results["kmeans"]["df_clustered"]
  or any similarly shaped DataFrame.
- Robust to missing years: only uses columns it finds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration & helpers
# -----------------------------------------------------------------------------

PURCHASE_PREFIX = "purchase_amount_eur_"
QTY_PREFIX = "quantity_sold_"
PRICE_PREFIX = "avg_price_eur_"
TURNOVER_PREFIX = "turnover_eur_"  # optional, used for context if present

DEFAULT_CLUSTER_COL = "kmeans_cluster"
ALT_CLUSTER_COLS = ("cluster", "cluster_id", "labels")

# sensible default palette for up to ~10 clusters
def _color_cycle(n: int) -> List[Tuple[float, float, float]]:
    cmap = plt.cm.get_cmap("tab10", max(n, 1))
    return [cmap(i) for i in range(n)]

def _ensure_cluster_col(df: pd.DataFrame) -> str:
    if DEFAULT_CLUSTER_COL in df.columns:
        return DEFAULT_CLUSTER_COL
    for c in ALT_CLUSTER_COLS:
        if c in df.columns:
            return c
    raise ValueError(
        f"No cluster column found. Expected one of: "
        f"{[DEFAULT_CLUSTER_COL]+list(ALT_CLUSTER_COLS)}"
    )

def _year_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    return sorted([c for c in df.columns if c.startswith(prefix) and c[len(prefix):].isdigit()])

def _as_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

@dataclass
class FeatureSpec:
    purch_cols: List[str]
    qty_cols: List[str]
    price_cols: List[str]

def _detect_features(df: pd.DataFrame) -> FeatureSpec:
    purch = _year_cols(df, PURCHASE_PREFIX)
    qty   = _year_cols(df, QTY_PREFIX)
    price = _year_cols(df, PRICE_PREFIX)
    return FeatureSpec(purch_cols=purch, qty_cols=qty, price_cols=price)

# -----------------------------------------------------------------------------
# Public: build derived totals (idempotent)
# -----------------------------------------------------------------------------

def build_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the following columns if the underlying year columns exist:
      - purchase_amount_eur_total
      - quantity_sold_total
      - avg_price_eur_overall  (volume-weighted if qty + price present; else mean)
      - year (long-form helper when melting)
    Safe to call multiple times.
    """
    spec = _detect_features(df)
    out = df.copy()
    out = _as_numeric(out, spec.purch_cols + spec.qty_cols + spec.price_cols)

    if spec.purch_cols and "purchase_amount_eur_total" not in out.columns:
        out["purchase_amount_eur_total"] = out[spec.purch_cols].sum(axis=1, skipna=True)

    if spec.qty_cols and "quantity_sold_total" not in out.columns:
        out["quantity_sold_total"] = out[spec.qty_cols].sum(axis=1, skipna=True)

    # weighted average price across years when possible
    if "avg_price_eur_overall" not in out.columns:
        if spec.price_cols and spec.qty_cols:
            # compute per-year revenue proxy = price * qty (when both present)
            common_years = sorted(
                set(int(c.split("_")[-1]) for c in spec.price_cols)
                & set(int(c.split("_")[-1]) for c in spec.qty_cols)
            )
            if common_years:
                price_cols = [f"{PRICE_PREFIX}{y}" for y in common_years]
                qty_cols   = [f"{QTY_PREFIX}{y}" for y in common_years]
                rev = (out[price_cols].values * out[qty_cols].values)
                rev_sum = np.nansum(rev, axis=1)
                qty_sum = np.nansum(out[qty_cols].values, axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    wavg = np.where(qty_sum > 0, rev_sum / qty_sum, np.nan)
                out["avg_price_eur_overall"] = wavg
            else:
                out["avg_price_eur_overall"] = np.nan
        elif spec.price_cols:
            out["avg_price_eur_overall"] = out[spec.price_cols].mean(axis=1, skipna=True)
        else:
            out["avg_price_eur_overall"] = np.nan

    return out

# -----------------------------------------------------------------------------
# Overall cluster metrics
# -----------------------------------------------------------------------------

def plot_overall_cluster_metrics(df: pd.DataFrame, show: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Bar charts of:
      - # items per cluster
      - total purchase_amount_eur_total per cluster
      - total quantity_sold_total per cluster
    Prints top/bottom clusters by each metric.
    """
    df2 = build_totals(df)
    clus_col = _ensure_cluster_col(df2)

    need = ["purchase_amount_eur_total", "quantity_sold_total"]
    missing = [c for c in need if c not in df2.columns]
    if missing:
        raise ValueError(f"Missing required totals: {missing}. Call build_totals() earlier.")

    # group metrics
    grp = (
        df2.groupby(clus_col, dropna=False)
           .agg(
               n_items=("item_number", "nunique") if "item_number" in df2.columns else ("purchase_amount_eur_total", "size"),
               purchase_eur_total=("purchase_amount_eur_total", "sum"),
               quantity_total=("quantity_sold_total", "sum"),
           )
           .sort_index()
    )

    print("📊 Overall cluster metrics")
    print(grp)

    if show:
        colors = _color_cycle(len(grp))
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
        x = np.arange(len(grp.index))

        axes[0].bar(x, grp["n_items"], color=colors)
        axes[0].set_title("# Items per cluster")
        axes[0].set_xlabel("Cluster"); axes[0].set_ylabel("Items")
        axes[0].set_xticks(x); axes[0].set_xticklabels(grp.index)

        axes[1].bar(x, grp["purchase_eur_total"], color=colors)
        axes[1].set_title("Total Purchase Amount (€)")
        axes[1].set_xlabel("Cluster"); axes[1].set_ylabel("€")
        axes[1].set_xticks(x); axes[1].set_xticklabels(grp.index)

        axes[2].bar(x, grp["quantity_total"], color=colors)
        axes[2].set_title("Total Quantity Sold")
        axes[2].set_xlabel("Cluster"); axes[2].set_ylabel("Units")
        axes[2].set_xticks(x); axes[2].set_xticklabels(grp.index)

        fig.suptitle("Overall metrics per cluster", fontsize=14, fontweight="bold")
        plt.show()

    # quick textual insights
    for col, label in (("purchase_eur_total", "purchase €"), ("quantity_total", "quantity"), ("n_items", "item count")):
        top = grp[col].idxmax()
        bot = grp[col].idxmin()
        print(f"• Top cluster by {label}: {top} (value={grp.loc[top, col]:,.0f}); "
              f"Lowest: {bot} (value={grp.loc[bot, col]:,.0f})")

    return {"cluster_overall": grp.reset_index()}

# -----------------------------------------------------------------------------
# Yearly trends (stacked by cluster + per-cluster line)
# -----------------------------------------------------------------------------

def plot_yearly_trends(df: pd.DataFrame, show: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Trends by year × cluster for:
      - purchase_amount_eur (sum)
      - quantity_sold (sum)
    """
    df2 = build_totals(df)
    clus_col = _ensure_cluster_col(df2)

    purch_cols = _year_cols(df2, PURCHASE_PREFIX)
    qty_cols   = _year_cols(df2, QTY_PREFIX)
    if not (purch_cols or qty_cols):
        raise ValueError("No year-suffixed purchase/quantity columns found.")

    long_parts = []
    if purch_cols:
        tmp = df2[[clus_col] + purch_cols].melt(id_vars=[clus_col], var_name="metric_year", value_name="value")
        tmp["metric"] = "purchase_eur"
        long_parts.append(tmp)
    if qty_cols:
        tmp = df2[[clus_col] + qty_cols].melt(id_vars=[clus_col], var_name="metric_year", value_name="value")
        tmp["metric"] = "quantity"
        long_parts.append(tmp)
    long = pd.concat(long_parts, ignore_index=True)
    long["year"] = long["metric_year"].str.extract(r"(\d{4})").astype(int)
    pivot = (
        long.groupby(["metric", "year", clus_col])["value"].sum().reset_index()
    )

    if show:
        colors_by_cluster = dict(zip(sorted(pivot[clus_col].unique()), _color_cycle(pivot[clus_col].nunique())))
        metrics = ["purchase_eur", "quantity"]
        titles = {"purchase_eur": "Purchase Amount (€)", "quantity": "Quantity Sold"}

        fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5), constrained_layout=True)
        if len(metrics) == 1:
            axes = [axes]

        for ax, m in zip(axes, metrics):
            sub = pivot[pivot["metric"] == m]
            for k in sorted(sub[clus_col].unique()):
                line = sub[sub[clus_col] == k].sort_values("year")
                ax.plot(line["year"], line["value"], marker="o", label=f"Cluster {k}",
                        color=colors_by_cluster[k])
            ax.set_title(titles[m]); ax.set_xlabel("Year"); ax.set_ylabel("Sum")
            ax.grid(True, alpha=.3); ax.legend()

        fig.suptitle("Yearly trends by cluster", fontsize=14, fontweight="bold")
        plt.show()

    return {"yearly_trends": pivot}

# -----------------------------------------------------------------------------
# Feature scatter + centroids
# -----------------------------------------------------------------------------

def plot_feature_scatter(
    df: pd.DataFrame,
    feature_x: str = "purchase_amount_eur_total",
    feature_y: str = "quantity_sold_total",
    centroids: Optional[np.ndarray] = None,
    show: bool = True
) -> Dict[str, float]:
    """
    2D scatter of the two clustering features colored by cluster.
    If you pass k-means centroids (scaled-back to original space), they'll be shown.
    Returns simple correlations for quick diagnostics.
    """
    df2 = build_totals(df)
    clus_col = _ensure_cluster_col(df2)
    need = [feature_x, feature_y]
    for c in need:
        if c not in df2.columns:
            raise ValueError(f"Missing feature column: {c}")

    # correlation diagnostics
    corr = float(pd.Series(df2[feature_x]).corr(pd.Series(df2[feature_y])))
    print(f"🔗 Pearson correlation between {feature_x} and {feature_y}: {corr:.3f}")

    if show:
        colors = dict(zip(sorted(df2[clus_col].unique()), _color_cycle(df2[clus_col].nunique())))
        fig, ax = plt.subplots(figsize=(7, 6))
        for k, sub in df2.groupby(clus_col):
            ax.scatter(sub[feature_x], sub[feature_y], s=20, alpha=0.6, label=f"Cluster {k}", color=colors[k])
        if centroids is not None and len(centroids) > 0:
            cx, cy = centroids[:, 0], centroids[:, 1]
            ax.scatter(cx, cy, s=200, marker="X", edgecolor="black", linewidth=1.2, label="Centroids")
        ax.set_xlabel(feature_x.replace("_", " ").title())
        ax.set_ylabel(feature_y.replace("_", " ").title())
        ax.set_title("Feature scatter by cluster")
        ax.grid(True, alpha=.3); ax.legend()
        plt.show()

    return {"pearson_corr": corr}

# -----------------------------------------------------------------------------
# Country-of-origin heatmap (ratio per row)
# -----------------------------------------------------------------------------

def plot_origin_heatmap(df: pd.DataFrame, min_count: int = 10, show: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Heatmap of STEP_CountryOfOrigin × cluster; values are row-normalized shares.
    min_count filters rare countries for a cleaner view.
    """
    if "STEP_CountryOfOrigin" not in df.columns:
        raise ValueError("STEP_CountryOfOrigin column not found.")
    clus_col = _ensure_cluster_col(df)
    sub = df[[clus_col, "STEP_CountryOfOrigin"]].copy()
    sub["STEP_CountryOfOrigin"] = sub["STEP_CountryOfOrigin"].fillna("Unknown")

    raw = sub.value_counts(["STEP_CountryOfOrigin", clus_col]).rename("n").reset_index()
    keep = raw.groupby("STEP_CountryOfOrigin")["n"].sum().loc[lambda s: s >= min_count].index
    raw = raw[raw["STEP_CountryOfOrigin"].isin(keep)]

    mat = raw.pivot(index="STEP_CountryOfOrigin", columns=clus_col, values="n").fillna(0.0)
    ratios = mat.div(mat.sum(axis=1), axis=0)

    if show:
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(ratios))))
        im = ax.imshow(ratios.values, aspect="auto", cmap="Blues", vmin=0, vmax=ratios.values.max() or 1)
        ax.set_yticks(range(len(ratios.index))); ax.set_yticklabels(ratios.index)
        ax.set_xticks(range(len(ratios.columns))); ax.set_xticklabels(ratios.columns)
        ax.set_xlabel("Cluster"); ax.set_ylabel("Country of Origin")
        ax.set_title("Country-of-origin share by cluster")
        for i in range(ratios.shape[0]):
            for j in range(ratios.shape[1]):
                ax.text(j, i, f"{ratios.iat[i, j]:.2f}", ha="center", va="center",
                        color="white" if ratios.iat[i, j] > 0.5 else "black", fontsize=8)
        cbar = plt.colorbar(im, ax=ax); cbar.set_label("Share within country")
        plt.tight_layout(); plt.show()

    return {"origin_counts": mat.reset_index(), "origin_ratios": ratios.reset_index()}

# -----------------------------------------------------------------------------
# Brand & Supplier distributions (top-N)
# -----------------------------------------------------------------------------

def _topN_by_cluster(df: pd.DataFrame, col: str, n: int = 10, other_label: str = "Other") -> Dict[int, pd.DataFrame]:
    clus_col = _ensure_cluster_col(df)
    out: Dict[int, pd.DataFrame] = {}
    for k, sub in df.groupby(clus_col):
        counts = sub[col].fillna("Unknown").value_counts(dropna=False)
        top = counts.head(n)
        remainder = counts.iloc[n:].sum()
        if remainder > 0:
            top = pd.concat([top, pd.Series({other_label: remainder})])
        out[int(k)] = top.rename("count").to_frame().reset_index(names=col)
    return out

def plot_brand_supplier_bars(
    df: pd.DataFrame,
    top_n: int = 8,
    show: bool = True
) -> Dict[str, Dict[int, pd.DataFrame]]:
    """
    Horizontal bars of top-N Brand & Supplier per cluster (with 'Other' bucket).
    """
    clus_col = _ensure_cluster_col(df)
    need_any = ("brand_name" in df.columns) or ("supplier_name_string" in df.columns)
    if not need_any:
        raise ValueError("Expected at least one of: brand_name, supplier_name_string.")

    figs = []
    results: Dict[str, Dict[int, pd.DataFrame]] = {}

    if "brand_name" in df.columns:
        res_b = _topN_by_cluster(df, "brand_name", n=top_n)
        results["brand"] = res_b
        if show:
            for k, tbl in res_b.items():
                fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(tbl))))
                ax.barh(tbl["brand_name"], tbl["count"], color="steelblue", alpha=.8)
                ax.set_title(f"Top {top_n} Brands • Cluster {k}")
                ax.set_xlabel("Items"); ax.grid(True, axis="x", alpha=.3)
                plt.tight_layout(); plt.show()
                figs.append(fig)

    if "supplier_name_string" in df.columns:
        res_s = _topN_by_cluster(df, "supplier_name_string", n=top_n)
        results["supplier"] = res_s
        if show:
            for k, tbl in res_s.items():
                fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(tbl))))
                ax.barh(tbl["supplier_name_string"], tbl["count"], color="darkorange", alpha=.8)
                ax.set_title(f"Top {top_n} Group Suppliers • Cluster {k}")
                ax.set_xlabel("Items"); ax.grid(True, axis="x", alpha=.3)
                plt.tight_layout(); plt.show()
                figs.append(fig)

    return results

# -----------------------------------------------------------------------------
# Compact “all-in” summary helper
# -----------------------------------------------------------------------------

def summarize_cluster_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Returns handy summary tables for export/reporting:
      - cluster_overall (size, total purchase €, total qty, avg price)
      - per_cluster_feature_quartiles (Q1/Median/Q3 for features)
    """
    df2 = build_totals(df)
    clus_col = _ensure_cluster_col(df2)
    base = (
        df2.groupby(clus_col, dropna=False)
           .agg(
               n_items=("item_number", "nunique") if "item_number" in df2.columns else ("purchase_amount_eur_total", "size"),
               purchase_eur_total=("purchase_amount_eur_total", "sum"),
               quantity_total=("quantity_sold_total", "sum"),
               avg_price_mean=("avg_price_eur_overall", "mean"),
           )
           .reset_index()
    )

    def _quartiles(s: pd.Series) -> Tuple[float, float, float]:
        q1, med, q3 = np.nanpercentile(s.values.astype(float), [25, 50, 75]) if len(s) else (np.nan, np.nan, np.nan)
        return float(q1), float(med), float(q3)

    rows = []
    for k, sub in df2.groupby(clus_col):
        q_purch = _quartiles(sub["purchase_amount_eur_total"])
        q_qty   = _quartiles(sub["quantity_sold_total"])
        rows.append({
            clus_col: k,
            "purch_q1": q_purch[0], "purch_median": q_purch[1], "purch_q3": q_purch[2],
            "qty_q1":   q_qty[0],   "qty_median":   q_qty[1],   "qty_q3":   q_qty[2],
        })
    quart = pd.DataFrame(rows)

    return {"cluster_overall": base, "per_cluster_feature_quartiles": quart}
