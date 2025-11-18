"""
analysis_utils.py

An integrated spend-analysis toolkit that wires the previously prepared
functions together **and** uses the attached helpers:
- field_desc_utils: field-name preparation & harmonization
- field_value_utils: value formatting/parsing + value-prep pipeline

This module composes:
- Tiering (ABC/k-means/GMM), summaries, dashboards
- BigQuery fetching (optional, env-driven)
- Robust preprocessing that **always** runs field-name/value prep first

Notes
-----
- If your environment doesn't have the helper modules on PYTHONPATH, we 
  dynamically load them from /mnt/data (where you attached them).
- Column access in analysis functions is robust to naming changes via
  `_resolve_col` which matches case/spacing/underscore and the preprocessed name.
"""
from __future__ import annotations

from typing import Tuple, Optional, Dict
import os
from decimal import Decimal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Load helper modules from attachments (with filesystem fallback)
# ---------------------------------------------------------------------------
try:
    from source.data_prep.field_desc_utils import (
        preprocess_field_name,
        preprocess_field_names,
        harmonize_field_names,
        prepare_field_names,
    )
except Exception:
    import importlib.util as _ils
    _p = "/mnt/data/field_desc_utils.py"
    _spec = _ils.spec_from_file_location("field_desc_utils", _p)
    _mod = _ils.module_from_spec(_spec) if _spec else None
    if _spec and _mod:
        _spec.loader.exec_module(_mod)  # type: ignore
        preprocess_field_name = _mod.preprocess_field_name
        preprocess_field_names = _mod.preprocess_field_names
        harmonize_field_names = _mod.harmonize_field_names
        prepare_field_names = _mod.prepare_field_names
    else:
        raise

try:
    from source.data_prep.field_value_utils import (
        fmt_eur,
        parse_eur,
        is_valid_eur_format,
        add_eur_suffix,
        fmt_units,
        parse_units,
        is_valid_units_format,
        preprocess_field_values,
        harmonize_field_values,
        prepare_field_values,
    )
except Exception:
    import importlib.util as _ils2
    _p2 = "/mnt/data/field_value_utils.py"
    _spec2 = _ils2.spec_from_file_location("field_value_utils", _p2)
    _mod2 = _ils2.module_from_spec(_spec2) if _spec2 else None
    if _spec2 and _mod2:
        _spec2.loader.exec_module(_mod2)  # type: ignore
        fmt_eur = _mod2.fmt_eur
        parse_eur = _mod2.parse_eur
        is_valid_eur_format = _mod2.is_valid_eur_format
        add_eur_suffix = _mod2.add_eur_suffix
        fmt_units = _mod2.fmt_units
        parse_units = _mod2.parse_units
        is_valid_units_format = _mod2.is_valid_units_format
        preprocess_field_values = _mod2.preprocess_field_values
        harmonize_field_values = _mod2.harmonize_field_values
        prepare_field_values = _mod2.prepare_field_values
    else:
        raise

# ---------------------------------------------------------------------------
# Optional sklearn + BigQuery dependencies
# ---------------------------------------------------------------------------
try:
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
except Exception:
    KMeans = None
    GaussianMixture = None

try:
    from source.db_connect.bigquery_connector import BigQueryConnector
    from source.db_connect.sql_queries import get_query
except Exception:
    BigQueryConnector = None
    get_query = None

from dotenv import load_dotenv
load_dotenv()
PROJECT_ID = os.getenv('PROJECT_ID')
DATASET_ID = os.getenv('DATASET_ID')
TABLE_ID = os.getenv('TABLE_ID')
if PROJECT_ID and DATASET_ID and TABLE_ID:
    PURCHASE_DATA_TABLE = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
else:
    PURCHASE_DATA_TABLE = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ABC_DEFAULT_THRESHOLDS: Tuple[float, float] = (0.80, 0.95)

def _normalize_key(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", s.lower())

def _resolve_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Resolve a column by trying multiple candidate names (robust).
    Tries raw candidates, their preprocessed variants, and case/spacing-insensitive matches.
    """
    if not candidates:
        raise ValueError("No candidates provided for column resolution.")
    norm_map = {_normalize_key(c): c for c in df.columns}
    for cand in candidates:
        # Exact
        if cand in df.columns:
            return cand
        # Normalized
        key = _normalize_key(cand)
        if key in norm_map:
            return norm_map[key]
        # Preprocessed variant
        pp = preprocess_field_name(cand)
        key2 = _normalize_key(pp)
        if key2 in norm_map:
            return norm_map[key2]
    # Fall back to first candidate; downstream code will error clearly if missing
    return candidates[0]

def _resolve_spend_column(df: pd.DataFrame, spend_col: str) -> tuple[pd.DataFrame, str]:
    """Ensure a spend column exists; derive if necessary from purchase_amount_eur * purchase_quantity."""
    from decimal import Decimal as _D
    if spend_col not in df.columns:
        # try common alternates
        alt = None
        for c in ['TotalSpend', 'total_spend', 'Total Spend']:
            if c in df.columns:
                alt = c
                break
        if alt:
            spend_col = alt
        elif {'purchase_amount_eur', 'purchase_quantity'} <= set(df.columns):
            amt = df['purchase_amount_eur'].apply(lambda x: float(x) if isinstance(x, _D) else x)
            qty = df['purchase_quantity'].apply(lambda x: float(x) if isinstance(x, _D) else x)
            df = df.copy()
            df[spend_col] = pd.to_numeric(amt, errors='coerce').fillna(0) * pd.to_numeric(qty, errors='coerce').fillna(0)
        else:
            raise ValueError(
                f"Spend column '{spend_col}' not found and cannot derive (missing purchase_amount_eur/purchase_quantity)."
            )
    return df, spend_col

# ---------------------------------------------------------------------------
# Internal preprocessing helpers (missing previously; now added)
# ---------------------------------------------------------------------------

def _drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename duplicate columns by appending __dupN suffix to ensure unique names.
    Keeps first occurrence unchanged; subsequent duplicates get incremental suffixes.
    """
    cols = list(df.columns)
    seen = {}
    new_cols = []
    dup_counts = {}
    changed = False
    for c in cols:
        if c not in seen:
            seen[c] = 1
            new_cols.append(c)
        else:
            dup_counts[c] = dup_counts.get(c, 0) + 1
            new_name = f"{c}__dup{dup_counts[c]}"
            # ensure uniqueness if somehow clashes
            while new_name in seen:
                dup_counts[c] += 1
                new_name = f"{c}__dup{dup_counts[c]}"
            new_cols.append(new_name)
            seen[new_name] = 1
            changed = True
    if changed:
        df = df.copy()
        df.columns = new_cols
        total_dups = sum(dup_counts.values())
        print(f"🔁 Renamed {total_dups} duplicate column(s).")
    return df

def _safe_apply_prepare_field_names(df: pd.DataFrame) -> pd.DataFrame:
    """Safely apply prepare_field_names; on failure, log and return original."""
    try:
        return prepare_field_names(df)
    except Exception as e:
        print(f"⚠️ prepare_field_names failed: {e}; continuing without name prep.")
        return df

# ---------------------------------------------------------------------------
# Data fetching (optional)
# ---------------------------------------------------------------------------

def fetch_purchase_data(bq_client: 'BigQueryConnector', limit: Optional[int] = None) -> pd.DataFrame:
    # Lazy import in case initial module import failed (e.g., due to transient encoding issues)
    global BigQueryConnector, get_query
    if BigQueryConnector is None or get_query is None:
        try:
            from source.db_connect.bigquery_connector import BigQueryConnector as _BQC
            from source.db_connect.sql_queries import get_query as _get_query
            BigQueryConnector = _BQC
            get_query = _get_query
        except Exception as e:
            raise ImportError("BigQuery dependencies not available after lazy import attempt: " + str(e))
    if not PURCHASE_DATA_TABLE:
        raise EnvironmentError("PROJECT_ID/DATASET_ID/TABLE_ID env vars are not set.")
    print("Fetching purchase data from BigQuery...")
    query = get_query('fetch_purchase_data').format(purchase_data_table=PURCHASE_DATA_TABLE)
    if limit is not None:
        query += f" LIMIT {limit}"
    df = bq_client.query(query)
    if df is None:
        raise RuntimeError("BigQuery query returned no results; check credentials, project, and table access.")
    return df

# ---------------------------------------------------------------------------
# Preprocessing: now uses attached helpers first
# ---------------------------------------------------------------------------

def preprocess_detailed_data(df: pd.DataFrame) -> pd.DataFrame:
    print("🔧 Preprocessing detailed data...")

    # 0) Deduplicate column names early
    df = _drop_duplicate_columns(df)

    # 1) Field-name prep (safe wrapper) + 2) Field-value prep
    df = _safe_apply_prepare_field_names(df)
    try:
        df = prepare_field_values(df)
    except Exception as e:
        print(f"⚠️ prepare_field_values failed: {e}; continuing without value prep.")

    original_len = len(df)

    # 3) Missing values (compact)
    print("📊 Checking for missing values...")
    null_counts = df.isnull().sum()
    for column, cnt in null_counts[null_counts > 0].items():
        print(f"  {column}: {cnt} missing values")

    # 4) Filter PLM status (if present under any normalized naming)
    for cand in ['PLMStatusGlobal', 'PLM Status Global', 'Plm Status Global']:
        if cand in df.columns:
            df = df[~df[cand].astype(str).str.contains(r'500|600|700', na=False)]
            print(f"Removed {original_len - len(df)} rows based on {cand} filter.")
            break
    else:
        print("ℹ️ No PLM status column found; skipping filter.")

    return df

# ---------------------------------------------------------------------------
# ABC / clustering tiers + summary
# ---------------------------------------------------------------------------

def compute_abc_tiers(
    df: pd.DataFrame,
    spend_col: str = 'total_spend',
    thresholds: Tuple[float, float] = ABC_DEFAULT_THRESHOLDS,
    ascending: bool = False,
    tier_col: str = 'abc_tier'
) -> pd.DataFrame:
    if len(thresholds) != 2 or not (0 < thresholds[0] < thresholds[1] < 1):
        raise ValueError("Thresholds must be (a_cut, b_cut) with 0 < a_cut < b_cut < 1.")
    df, spend_col = _resolve_spend_column(df, spend_col)
    work = df[[spend_col]].copy()
    spend_values = work[spend_col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
    work['spend'] = pd.to_numeric(spend_values, errors='coerce').fillna(0)
    ranked = work.loc[work['spend'].sort_values(ascending=ascending).index].copy()
    total = ranked['spend'].sum()
    if total <= 0:
        raise ValueError("Total spend is zero; cannot compute ABC tiers.")
    ranked['cum_share'] = ranked['spend'].cumsum() / total
    a_cut, b_cut = thresholds
    def _assign(cum: float) -> str:
        if cum <= a_cut:
            return 'A'
        if cum <= b_cut:
            return 'B'
        return 'C'
    ranked[tier_col] = ranked['cum_share'].apply(_assign)
    out = df.copy()
    out[spend_col] = out[spend_col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
    out[tier_col] = ranked[tier_col].reindex(out.index)
    return out

def compute_kmeans_tiers(
    df: pd.DataFrame,
    spend_col: str = 'total_spend',
    k: int = 3,
    log_transform: bool = True,
    tier_col: str = 'kmeans_tier'
) -> pd.DataFrame:
    if KMeans is None:
        raise ImportError("sklearn not available for k-means tiering.")
    if k < 2 or k > 6:
        raise ValueError("k should be between 2 and 6 for interpretability.")
    df, spend_col = _resolve_spend_column(df, spend_col)
    X = pd.to_numeric(df[spend_col], errors='coerce').fillna(0).values.reshape(-1, 1)
    X = np.log1p(X) if log_transform else X
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_.flatten()
    order = np.argsort(centers)[::-1]
    tier_map: Dict[int, str] = {cluster_id: "ABCDEF"[i] for i, cluster_id in enumerate(order)}
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
    if GaussianMixture is None:
        raise ImportError("sklearn not available for GMM tiering.")
    if max_components < 2 or max_components > 6:
        raise ValueError("max_components should be between 2 and 6.")
    df, spend_col = _resolve_spend_column(df, spend_col)
    X0 = pd.to_numeric(df[spend_col], errors='coerce').fillna(0).values
    X = np.log1p(X0).reshape(-1, 1) if log_transform else X0.reshape(-1, 1)
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
    tier_map: Dict[int, str] = {comp_id: "ABCDEF"[i] for i, comp_id in enumerate(order)}
    out = df.copy()
    out[tier_col] = [tier_map[l] for l in labels]
    return out

def summarize_tiers(df: pd.DataFrame, spend_col: str, tier_col: str) -> pd.DataFrame:
    df, spend_col = _resolve_spend_column(df, spend_col)
    spend_numeric = pd.to_numeric(df[spend_col].apply(lambda x: float(x) if isinstance(x, Decimal) else x), errors='coerce').fillna(0)
    df = df.copy()
    df[spend_col] = spend_numeric
    total = spend_numeric.sum()
    grp = df.groupby(tier_col)[spend_col].agg(['count', 'sum']).rename(columns={'sum': 'spend'})
    grp['share_pct'] = (grp['spend'] / total * 100.0) if total else 0.0
    return grp.reset_index().sort_values('share_pct', ascending=False)

# ---------------------------------------------------------------------------
# Visualizations (use robust column resolution)
# ---------------------------------------------------------------------------

def create_dashboard_distributions(df: pd.DataFrame, columns_to_plot, outlier_columns=['q95_quantity', 'total_purchase_amount_eur']):
    print("📊 Creating dashboard for distributions...")
    for col in outlier_columns:
        if col in df.columns:
            threshold = df[col].quantile(0.95)
            df = df[df[col] <= threshold]
            print(f"  Removed outliers in {col} above {threshold}")
    num_plots = len(columns_to_plot)
    cols = 3
    rows = (num_plots + cols - 1) // cols
    plt.figure(figsize=(5 * cols, 4 * rows))
    for i, column in enumerate(columns_to_plot):
        plt.subplot(rows, cols, i + 1)
        sns.histplot(df[column].dropna(), kde=True, bins=30)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
    plt.tight_layout(); plt.show()

def create_class3_countryoforigin_relationship_dashboard(df: pd.DataFrame, columns_to_plot, outlier_columns=['q95_quantity', 'total_purchase_amount_eur']):
    print("📊 Creating Class3 vs Country of Origin relationship dashboard...")
    for col in outlier_columns:
        if col in df.columns:
            threshold = df[col].quantile(0.95)
            df = df[df[col] <= threshold]
            print(f"  Removed outliers in {col} above {threshold}")
    num_plots = len(columns_to_plot)
    cols = 3
    rows = (num_plots + cols - 1) // cols
    plt.figure(figsize=(5 * cols, 4 * rows))
    class3_col = _resolve_col(df, ['Class3', 'class3', 'Class 3'])
    for i, column in enumerate(columns_to_plot):
        plt.subplot(rows, cols, i + 1)
        sns.boxplot(x=class3_col, y=column, data=df)
        plt.title(f'{column} by Class3')
        plt.xlabel('Class3'); plt.ylabel(column); plt.xticks(rotation=45)
    plt.tight_layout(); plt.show()

def analyze_spend_by_group_vendor(df: pd.DataFrame):
    print("📊 Analyzing spend by Group Vendor...")
    vendor_col = _resolve_col(df, ['GroupVendor', 'crm_main_group_vendor', 'Group Vendor'])
    spend_col = _resolve_col(df, ['TotalSpend', 'total_spend', 'Total Spend'])
    spend_by_vendor = df.groupby(vendor_col)[spend_col].sum().reset_index()
    top_vendors = spend_by_vendor.sort_values(by=spend_col, ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=spend_col, y=vendor_col, data=top_vendors)
    plt.title('Top 10 Group Vendors by Total Spend')
    plt.xlabel('Total Spend'); plt.ylabel('Group Vendor')
    plt.tight_layout(); plt.show()
    print("Basic statistics of Total Spend by Group Vendor:")
    print(spend_by_vendor[spend_col].describe())

def analyze_spend_trends_by_group_vendor_and_countryoforigin(df: pd.DataFrame):
    print("📊 Analyzing spend trends by Group Vendor and Country of Origin...")
    vendor_col = _resolve_col(df, ['GroupVendor', 'crm_main_group_vendor', 'Group Vendor'])
    origin_col = _resolve_col(df, ['CountryOfOrigin', 'country_of_origin', 'Country Of Origin'])
    spend_col = _resolve_col(df, ['TotalSpend', 'total_spend', 'Total Spend'])
    spend_trends = df.groupby([vendor_col, origin_col])[spend_col].sum().reset_index()
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=vendor_col, y=spend_col, hue=origin_col, data=spend_trends, s=100)
    plt.title('Spend Trends by Group Vendor and Country of Origin')
    plt.xlabel('Group Vendor'); plt.ylabel('Total Spend')
    plt.xticks(rotation=45); plt.legend(title='Country of Origin', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(); plt.show()

def analyze_spend_by_productnumber(df: pd.DataFrame):
    print("📊 Analyzing spend by Product Number...")
    prod_col = _resolve_col(df, ['ProductNumber', 'product_number', 'Product Number'])
    spend_col = _resolve_col(df, ['TotalSpend', 'total_spend', 'Total Spend'])
    spend_by_product = df.groupby(prod_col)[spend_col].sum().reset_index()
    top_products = spend_by_product.sort_values(by=spend_col, ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=spend_col, y=prod_col, data=top_products)
    plt.title('Top 10 Products by Total Spend')
    plt.xlabel('Total Spend'); plt.ylabel('Product Number')
    plt.tight_layout(); plt.show()
    print("Basic statistics of Total Spend by Product Number:")
    print(spend_by_product[spend_col].describe())

def abc_segmentation_analysis(df: pd.DataFrame, spend_col: str = 'total_spend', thresholds: Tuple[float, float] = ABC_DEFAULT_THRESHOLDS, tier_col: str = 'abc_tier') -> pd.DataFrame:
    print("📊 Performing ABC segmentation analysis...")
    df_tiered = compute_abc_tiers(df, spend_col=spend_col, thresholds=thresholds, tier_col=tier_col)
    plt.figure(figsize=(10, 6))
    sns.countplot(x=tier_col, data=df_tiered, order=['A', 'B', 'C'])
    plt.title('ABC Segmentation of Items'); plt.xlabel('ABC Tier'); plt.ylabel('Number of Items')
    plt.tight_layout(); plt.show()
    return df_tiered

# Lightweight helpers

def analyze_spend_distribution(df: pd.DataFrame, spend_col: str = 'total_spend', top_n: int = 15) -> None:
    df, spend_col = _resolve_spend_column(df, spend_col)
    df = df.copy(); df[spend_col] = pd.to_numeric(df[spend_col], errors='coerce').fillna(0)
    plt.figure(figsize=(10, 5))
    sns.histplot(df[spend_col], bins=50, kde=True)
    plt.title('Spend Distribution'); plt.xlabel('Spend'); plt.ylabel('Frequency')
    plt.tight_layout(); plt.show()
    top_df = df.nlargest(top_n, spend_col)
    print(f"Top {top_n} items by {spend_col}:")
    print(top_df[[spend_col]].head(top_n))

def analyze_supplier_distribution(df: pd.DataFrame, supplier_col: str = 'crm_main_vendor', spend_col: str = 'total_spend', top_n: int = 15) -> None:
    if supplier_col not in df.columns:
        fallback = _resolve_col(df, ['crm_main_group_vendor', 'Group Vendor', 'Vendor'])
        print(f"ℹ️ '{supplier_col}' not found; using '{fallback}' instead.")
        supplier_col = fallback
    df, spend_col = _resolve_spend_column(df, spend_col)
    df = df.copy(); df[spend_col] = pd.to_numeric(df[spend_col], errors='coerce').fillna(0)
    grouped = df.groupby(supplier_col)[spend_col].sum().sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=grouped.values, y=grouped.index)
    plt.title(f'Top {top_n} Suppliers by Spend'); plt.xlabel('Total Spend'); plt.ylabel('Supplier')
    plt.tight_layout(); plt.show()
    print(grouped)

def analyze_purchase_frequency(df: pd.DataFrame, purchase_col: str = 'purchase_quantity') -> None:
    if purchase_col not in df.columns:
        print(f"⚠️ Purchase frequency column '{purchase_col}' not found; skipping frequency analysis.")
        return
    vals = pd.to_numeric(df[purchase_col], errors='coerce').dropna()
    plt.figure(figsize=(10, 5))
    sns.histplot(vals, bins=50, kde=True)
    plt.title('Purchase Quantity Frequency'); plt.xlabel('Quantity'); plt.ylabel('Frequency')
    plt.tight_layout(); plt.show()
    print(vals.describe())

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_results_per_class3(
    df: pd.DataFrame,
    tier_col: str = 'abc_tier',
    output_dir: Optional[str] = None
) -> None:
    import os as _os
    if output_dir is None:
        output_dir = _os.getcwd()
    class3_col = _resolve_col(df, ['Class3', 'class3', 'Class 3'])
    if class3_col not in df.columns:
        raise KeyError("'Class3' column required for export.")
    class3_values = df[class3_col].dropna().unique()
    for class3 in class3_values:
        subset = df[df[class3_col] == class3]
        filename = f"ABC_Segmentation_{str(class3).replace('/', '_')}.xlsx"
        filepath = _os.path.join(output_dir, filename)
        subset.to_excel(filepath, index=False)
        print(f"✅ Exported {filepath} with {len(subset)} records.")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    # Helper re-exports
    'preprocess_field_name','preprocess_field_names','harmonize_field_names','prepare_field_names',
    'fmt_eur','parse_eur','is_valid_eur_format','add_eur_suffix','fmt_units','parse_units','is_valid_units_format',
    'preprocess_field_values','harmonize_field_values','prepare_field_values',
    # Fetch & preprocess
    'fetch_purchase_data','preprocess_detailed_data',
    # Tiering
    'compute_abc_tiers','compute_kmeans_tiers','compute_gmm_tiers','summarize_tiers',
    # Analyses / viz
    'create_dashboard_distributions','create_class3_countryoforigin_relationship_dashboard',
    'analyze_spend_by_group_vendor','analyze_spend_trends_by_group_vendor_and_countryoforigin',
    'analyze_spend_by_productnumber','abc_segmentation_analysis',
    'analyze_spend_distribution','analyze_supplier_distribution','analyze_purchase_frequency',
    # Export
    'export_results_per_class3',
]
