"""
Centralized Parameters for Tender Material / Clustering Analysis

This module centralizes all configuration used by notebooks and utilities:
- Data source (EU): kramp-sharedmasterdata-prd.MadsH.tender_material
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
TABLE_ID:   str = os.getenv("TABLE_ID", "tender_material")

# Fully-qualified name used by query helpers
TENDER_MATERIAL_FQN: str = f"`{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`"

# BigQuery job location – keep EU to avoid cross-region issues
BQ_LOCATION: str = os.getenv("BQ_LOCATION", "EU")

# Optional row limit for quick iterations (None for full data)
ROW_LIMIT: Optional[int] = None  # e.g., 50_000


# =============================================================================
# DATA FILTERING PARAMETERS (SQL-side or DF-side)
# =============================================================================
# These map 1:1 to columns in MadsH.tender_material and to tender_material_utils.fetch_tender_material()

# Purchase-stop handling:
#   'strict'  -> keep only stop_purchase_ind = 'N'
#   'lenient' -> keep rows where stop_purchase_ind != 'Y' OR IS NULL
#   None      -> no purchase-stop filter
PURCHASE_STOP_MODE: Optional[str] = "strict"

# Exact-match filters (None or empty list => no filter)
CLASS2: Optional[Sequence[str]] = ["Fasteners"]        # ex: ["Fasteners"]
CLASS3: Optional[Sequence[str]] = ["Threaded Rods"]    # ex: ["Threaded Rods"]
CLASS4: Optional[Sequence[str]] = None                 # ex: ["1234 - Filters"]
BRAND_NAME: Optional[Sequence[str]] = ["Kramp"]        # ex: ["Kramp", "Kubota"]

# New filters you added:
COUNTRY_OF_ORIGIN: Optional[Sequence[str]] = None      # column STEP_CountryOfOrigin, e.g., ["Germany","Italy"]
GROUP_SUPPLIER: Optional[Sequence[str]] = None         # column supplier_name_string, e.g., ["Kerbl Group"]

# Other optional filters
ITEM_NUMBERS: Optional[Sequence[str]] = None           # item_number filter
ATTRIBUTE_IDS: Optional[Sequence[str]] = None          # attribute_id filter
KEYWORDS_IN_DESC: Optional[Sequence[str]] = None       # LIKE search on item_description (case-insensitive)
NEGATE_FILTERS: bool = False                           # invert filters (rare)

# Minimum transactions per product (for downstream analysis steps)
MIN_TRANSACTIONS: int = 50

# Optional rounding-value focus (if your analysis merges in sales/rounding later)
ROUNDING_FILTER: Optional[float] = 1.0  # None for all


# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

# Include product dimensions in clustering features (if you derive them later)
INCLUDE_DIMENSIONS: bool = True

# Attribute handling (from ciske attributes already joined in tender_material)
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
    Spec for building clustering features from tender_material columns.

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
# HELPER: Build a kwargs dict for tender_material_utils.fetch_tender_material()
# =============================================================================

def fetch_kwargs() -> dict:
    """
    Compose keyword arguments for tender_material_utils.fetch_tender_material()
    based on the centralized parameters above.
    """
    return {
        "columns": None,                      # None -> default columns from helper
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
