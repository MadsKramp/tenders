
from __future__ import annotations
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

from typing import Tuple, Optional, Dict
import os
from decimal import Decimal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
__all__ = [
    'preprocess_detailed_data',
    'make_spend_col',
    'compute_abc_tiers',
    'fetch_purchase_data',
    # add other public functions as needed
]
# ---------------------------------------------------------------------------
# Spend column normalization utility
# ---------------------------------------------------------------------------
def make_spend_col(df: pd.DataFrame, prefer: str = 'Purchase Amount Eur') -> tuple[pd.DataFrame, str]:
    """
    Ensures a numeric spend column exists and returns (df, spend_col).
    Prefers cleaned numeric columns (e.g., 'Purchase Amount Eur') if present.
    Prints diagnostics after conversion.
    """
    candidates = [
        prefer,
        'Purchase Amount Eur',
        'purchase_amount_eur',
        'PurchaseAmountEUR',
        'purchase_amount',
        'amount_eur',
        'AmountEUR',
        'amount',
        'total_spend',
        'TotalSpend',
        'Total Spend'
        'purchase_amount_eur'
    ]
    col = None
    for c in candidates:
        if c in df.columns:
            col = c
            break
    if col is None:
        raise ValueError("No spend column found in dataset.")
    out = df.copy()
    out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    print(f"make_spend_col: Using spend column '{col}' | non-null count: {out[col].notnull().sum()} | stats:")
    print(out[col].describe())
    return out, col
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
    # Only remove non-alphanumeric (but keep numbers)
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
        # Patch: ensure prepare_field_names does not remove numbers from field names
        # If the helper function does, override it here
        df2 = prepare_field_names(df)
        # Check for numeric loss in column names
        orig_cols = list(df.columns)
        new_cols = list(df2.columns)
        for o, n in zip(orig_cols, new_cols):
            # If original had a digit and new does not, revert to original
            if any(c.isdigit() for c in o) and not any(c.isdigit() for c in n):
                print(f"⚠️ Field name '{o}' lost digits in preprocessing; restoring original name.")
                df2 = df2.rename(columns={n: o})
        return df2
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
            raise ImportError(f"BigQuery dependencies not available after lazy import attempt: {e}")
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

    return df

def compute_abcde_per_class4(
    df,
    class_col,
    product_col,
    spend_col,
    qty_col=None,
    tiers=("A","B","C","D","E"),
    tier_thresholds=(0.8,0.95,0.99,0.999),
    min_products_per_tier=5,
    verbose=False
):
    """
    ABCDE segmentation per class_col based on cumulative share of spend_col.

    tier_thresholds define cumulative-spend cut-offs for each tier:
        e.g. [0.8, 0.95, 0.99, 0.999] → A≤80%, B≤95%, C≤99%, D≤99.9%, E>99.9%
    """
    import pandas as pd
    df = df.copy()
    tiers = list(tiers)
    if len(tiers) != len(tier_thresholds)+1:
        raise ValueError("tier_thresholds must have one fewer element than tiers")

    grouped = []
    for c3, g in df.groupby(class_col):
        g = g.sort_values(spend_col, ascending=False)
        total = g[spend_col].sum()
        if total <= 0:
            g["Segmentation"] = tiers[-1]
        else:
            g["cum_share"] = g[spend_col].cumsum() / total
            thresholds = [0.0] + list(tier_thresholds) + [1.0]
            g["Segmentation"] = pd.cut(
                g["cum_share"],
                bins=thresholds,
                labels=tiers,
                include_lowest=True,
                right=True
            ).astype(str)
        # guarantee at least min_products_per_tier
        if min_products_per_tier:
            for t in tiers:
                mask = g["Segmentation"] == t
                if mask.sum() < min_products_per_tier:
                    g.loc[mask, "Segmentation"] = tiers[-1]
        grouped.append(g)
        if verbose:
            print(f"{c3}: {g['Segmentation'].value_counts().to_dict()}")
    return pd.concat(grouped, ignore_index=True)


# ---------------------------------------------------------------------------
# Lightweight helpers

def analyze_spend_distribution(df: pd.DataFrame, spend_col: str = 'Purchase Amount Eur', top_n: int = 15) -> None:
    df, spend_col = _resolve_spend_column(df, spend_col)
    df = df.copy(); df[spend_col] = pd.to_numeric(df[spend_col], errors='coerce').fillna(0)
    plt.figure(figsize=(10, 5))
    sns.histplot(df[spend_col], bins=50, kde=True)
    plt.title('Spend Distribution'); plt.xlabel('Spend'); plt.ylabel('Frequency')
    plt.tight_layout(); plt.show()
    top_df = df.nlargest(top_n, spend_col)
    print(f"Top {top_n} items by {spend_col}:")
    print(top_df[[spend_col]].head(top_n))

def analyze_supplier_distribution(df: pd.DataFrame, supplier_col: str = 'crm_main_group_vendor', spend_col: str = 'Purchase Amount Eur', top_n: int = 15) -> None:
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

def analyze_purchase_frequency(df: pd.DataFrame, purchase_col: str = 'Purchase Quantity') -> None:
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
    'create_dashboard_distributions','create_class4_countryoforigin_relationship_dashboard',
    'analyze_spend_by_group_vendor','analyze_spend_trends_by_group_vendor_and_countryoforigin',
    'analyze_spend_by_productnumber','abc_segmentation_analysis',
    'analyze_spend_distribution','analyze_supplier_distribution','analyze_purchase_frequency',
    
]
