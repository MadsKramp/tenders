"""
Centralized Parameters for Tender Material / Clustering Analysis

This module centralizes all configuration used by notebooks and utilities:
- Data source (EU): kramp-sharedmasterdata-prd.MadsH.super_table
- SQL-side filtering toggles and values (Class2/3/4, Brand, Country of Origin, Group Supplier)
- Purchase-stop policy
- Feature toggles (attributes pivoting, dimensions)
- Clustering parameters and visualization/export options

Clustering features are derived ONLY from purchase_amount_eur and quantity_sold.
"""

from __future__ import annotations
import os
from typing import Optional, Sequence, Dict, List

# =============================================================================
# DATA SOURCE (EU)
# =============================================================================

PROJECT_ID: str = os.getenv("PROJECT_ID", "kramp-sharedmasterdata-prd")
DATASET_ID: str = os.getenv("DATASET_ID", "MadsH")
# Updated default to point at the denormalized super table (override via env if needed)
TABLE_ID:   str = os.getenv("TABLE_ID", "super_table")

# Sanitize any accidental fully-qualified strings in env vars (defensive):
def _sanitize_project(pid: str) -> str:
    # If someone passed 'proj.dataset.table', keep only first segment as project id.
    return pid.split('.')[0] if '.' in pid else pid

def _sanitize_dataset(did: str) -> str:
    # If dataset mistakenly includes table (e.g. 'dataset.table'), take first segment.
    return did.split('.')[0] if '.' in did else did

def _sanitize_table(tid: str) -> str:
    # If table mistakenly includes dataset (e.g. 'dataset.table'), take last segment.
    return tid.split('.')[-1] if '.' in tid else tid

PROJECT_ID = _sanitize_project(PROJECT_ID)
DATASET_ID = _sanitize_dataset(DATASET_ID)
TABLE_ID   = _sanitize_table(TABLE_ID)

# Fully-qualified name used by query helpers
SUPER_TABLE_FQN: str = f"`{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`"  # canonical FQN

# BigQuery job location – keep EU to avoid cross-region issues
BQ_LOCATION: str = os.getenv("BQ_LOCATION", "EU")

# Optional row limit for quick iterations (None for full data)
ROW_LIMIT: Optional[int] = None  # e.g., 50_000


# =============================================================================
# DATA FILTERING PARAMETERS (aligned to analysis_utils.fetch_super_table_filtered)
# =============================================================================
# Minimal filter surface: Class2, Class3, Brand, GroupVendorName (via purchase_data), description regex.

# Exact match filters (None/empty => no filter)
CLASS2: Optional[Sequence[str]] = ["Fasteners"]          # maps to column class_2
CLASS3: Optional[Sequence[str]] = ["Threaded Rods"]      # maps to column class_3
BRAND_NAME: Optional[Sequence[str]] = ["Kramp"]          # maps to column brand_name
GROUP_VENDOR_NAME: Optional[Sequence[str]] = None         # maps to purchase_data.group_vendor_dim.GroupVendorName

# Regex keyword patterns (OR combined) matched against item_description (case-insensitive)
DESCRIPTION_REGEX: Optional[Sequence[str]] = None         # e.g. ["stainless", "hex"]
REGEX_WHOLE_WORD: bool = True                             # wrap each pattern in \b...\b if True

# Legacy / deprecated filters retained for compatibility (not used by simplified fetch)
PURCHASE_STOP_MODE: Optional[str] = "strict"             # still consumed by broader fetch if used elsewhere
CLASS4: Optional[Sequence[str]] = None                    # not used in simplified filtered fetch
COUNTRY_OF_ORIGIN: Optional[Sequence[str]] = None         # not used in simplified filtered fetch
GROUP_SUPPLIER: Optional[Sequence[str]] = None            # superseded by GROUP_VENDOR_NAME
ITEM_NUMBERS: Optional[Sequence[str]] = None              # not used
ATTRIBUTE_IDS: Optional[Sequence[str]] = None             # not used
KEYWORDS_IN_DESC: Optional[Sequence[str]] = None          # superseded by DESCRIPTION_REGEX
NEGATE_FILTERS: bool = False                              # rarely needed

# Minimum transactions per product (for downstream analysis steps)
MIN_TRANSACTIONS: int = 50

# Optional rounding-value focus (if your analysis merges in sales/rounding later)
ROUNDING_FILTER: Optional[float] = 1.0  # None for all


# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

# Include product dimensions in clustering features (if you derive them later)
INCLUDE_DIMENSIONS: bool = True

# Attribute handling (from ciske attributes already joined in super_table)
INCLUDE_ATTRIBUTES_WIDE: bool = True
ATTR_VALUE_COLUMN: str = "locale_independent_values"  # or "localized_values"
ATTR_COLUMN_NAME: str = "attribute_id"
ATTR_WIDE_PREFIX: str = "attr_"
ATTR_FIRST_VALUE_ONLY: bool = True  # if multiple rows per (item_number, attribute_id), keep first

# Base product columns commonly selected for analysis/exports (subset is fine)
BASE_PRODUCT_COLUMNS = [
    "item_number",
    "item_description",
    "class_2", "class_3", "class_4",
    "brand_name", "brand_type", "brand_type_agg",
    "countries_webshop_ind_active", "count_countries_webshop_active",
    "stop_purchase_ind",
    "STEP_CountryOfOrigin",         # new
    "supplier_name_string",         # new (Group Supplier)
]


# =============================================================================
# CLUSTERING PARAMETERS
# =============================================================================

# ---- Core decision: cluster ONLY on purchase_amount_eur and quantity_sold ----
# Choose how to derive each from year-suffixed columns: "sum", "latest", or "mean"
CLUSTER_FEATURE_STRATEGY: str = "sum"           # default: sum across years
CLUSTER_FEATURE_YEARS: Sequence[int] = (2021, 2022, 2023, 2024)

# Names for the two numeric features that downstream code will consume
CLUSTER_FEATURES: Sequence[str] = (
    "feat_purchase_amount_eur",
    "feat_quantity_sold",
)

def feature_spec() -> Dict[str, Dict[str, List[str] | str]]:
    """
    Spec for building clustering features from super_table columns.

    Returns
    -------
    dict: {
      "feat_purchase_amount_eur": {"from": [...year columns...], "strategy": "sum|latest|mean"},
      "feat_quantity_sold":      {"from": [...year columns...], "strategy": "sum|latest|mean"},
    }
    """
    purch_cols = [f"purchase_amount_eur_{y}" for y in CLUSTER_FEATURE_YEARS]
    qty_cols   = [f"quantity_sold_{y}"        for y in CLUSTER_FEATURE_YEARS]
    return {
        "feat_purchase_amount_eur": {"from": purch_cols, "strategy": CLUSTER_FEATURE_STRATEGY},
        "feat_quantity_sold":      {"from": qty_cols,   "strategy": CLUSTER_FEATURE_STRATEGY},
    }

# Range of clusters to try when optimizing
MIN_CLUSTERS: int = 2
MAX_CLUSTERS: int = 10

# If you want a fixed number (bypass optimization), set this; else None
N_CLUSTERS_OVERRIDE: Optional[int] = None

# Random seed for reproducibility
RANDOM_STATE: int = 42

# Which clustering methods to run
INCLUDE_KMEANS: bool = True
INCLUDE_HIERARCHICAL: bool = True
INCLUDE_DBSCAN: bool = True

# DBSCAN specifics (None => let your pipeline auto-tune)
DBSCAN_EPS: Optional[float] = None
DBSCAN_MIN_SAMPLES: Optional[int] = None


# =============================================================================
# OUTPUT & VISUALIZATION
# =============================================================================

# Toggle charting in notebooks
SHOW_VISUALIZATIONS: bool = True

# Export results (e.g., CSV/Excel) in notebooks or pipelines
EXPORT_RESULTS: bool = True

# Output folder for artifacts (notebooks can respect this)
OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "artifacts")


# =============================================================================
# HELPER: Build a kwargs dict for super_table_utils.fetch_super_table()
# =============================================================================

def fetch_kwargs() -> dict:
    """Backward-compatible kwargs for the original broad fetch_super_table helper.

    Prefer ``filtered_fetch_kwargs()`` for the new minimal surface.
    """
    return {
        "columns": None,
        "class2": CLASS2,
        "class3": CLASS3,
        "class4": CLASS4,
        "brand_name": BRAND_NAME,
        "country_of_origin": COUNTRY_OF_ORIGIN,
        "group_supplier": GROUP_SUPPLIER,
        "item_numbers": ITEM_NUMBERS,
        "attribute_ids": ATTRIBUTE_IDS,
        "keyword_in_desc": KEYWORDS_IN_DESC,
        "purchase_stop": PURCHASE_STOP_MODE,
        "limit": ROW_LIMIT,
        "debug_print_sql": False,
    }

def filtered_fetch_kwargs() -> dict:
    """Kwargs for analysis_utils.fetch_super_table_filtered (recommended)."""
    return {
        "class2": CLASS2,
        "class3": CLASS3,
        "brand": BRAND_NAME,
        "group_vendor_name": GROUP_VENDOR_NAME,
        "description_regex": DESCRIPTION_REGEX,
        "regex_whole_word": REGEX_WHOLE_WORD,
        "limit": ROW_LIMIT,
        "debug_print_sql": False,
    }

__all__ = [
    # data source constants
    "PROJECT_ID", "DATASET_ID", "TABLE_ID", "SUPER_TABLE_FQN", "BQ_LOCATION",
    # filters (new minimal + legacy)
    "CLASS2", "CLASS3", "BRAND_NAME", "GROUP_VENDOR_NAME",
    "DESCRIPTION_REGEX", "REGEX_WHOLE_WORD",
    "PURCHASE_STOP_MODE", "CLASS4", "COUNTRY_OF_ORIGIN", "GROUP_SUPPLIER",
    "ITEM_NUMBERS", "ATTRIBUTE_IDS", "KEYWORDS_IN_DESC", "NEGATE_FILTERS",
    # clustering feature config
    "CLUSTER_FEATURE_STRATEGY", "CLUSTER_FEATURE_YEARS", "CLUSTER_FEATURES", "feature_spec",
    # clustering knobs
    "MIN_CLUSTERS", "MAX_CLUSTERS", "N_CLUSTERS_OVERRIDE", "RANDOM_STATE",
    "INCLUDE_KMEANS", "INCLUDE_HIERARCHICAL", "INCLUDE_DBSCAN", "DBSCAN_EPS", "DBSCAN_MIN_SAMPLES",
    # output settings
    "SHOW_VISUALIZATIONS", "EXPORT_RESULTS", "OUTPUT_DIR",
    # helpers
    "fetch_kwargs", "filtered_fetch_kwargs",
]
