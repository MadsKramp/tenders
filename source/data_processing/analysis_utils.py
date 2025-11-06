"""
super_table_utils.py — Utilities for the MadsH.super_table table (EU)

- BigQuery fetches from: kramp-sharedmasterdata-prd.MadsH.super_table
- Optional filters: Class2, Class3, Class4, BrandName, CountryOfOrigin, GroupSupplier
- Purchase-stop filtering: strict (stop_purchase_ind='N'),
  lenient (stop_purchase_ind!='Y' OR NULLs), or off.
- Helpers to pivot attribute rows to wide columns.

Dependencies expected in your repo:
- BigQueryConnector at source.db_connect.BigQueryConnector
- product_utils with filter helpers (we use them optionally for local DF filtering)

Tip: Configure PROJECT_ID/DATASET_ID/TABLE_ID via env vars; defaults included.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence

import pandas as pd

from dotenv import load_dotenv

# Your existing connector
from source.db_connect import BigQueryConnector

# Optional: use your dataframe-side helpers when you already have a DF loaded
# (not required for SQL-side filtering, but nice to keep parity with the notebook)
try:
    from source.data_processing.product_utils import (
        filter_by_class2,
        filter_by_class3,
        filter_by_class4,
        filter_by_brand_name,
        filter_by_country_of_origin,
        filter_by_group_supplier,
        filter_not_purchase_stop,  # dataframe-side purchase-stop
        coerce_dtypes,
        validate_has,
    )
except Exception:  # pragma: no cover
    filter_by_class2 = filter_by_class3 = filter_by_class4 = filter_by_brand_name = None
    filter_by_country_of_origin = filter_by_group_supplier = None
    filter_not_purchase_stop = None
    coerce_dtypes = validate_has = None

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID", "kramp-sharedmasterdata-prd")
DATASET_ID = os.getenv("DATASET_ID", "MadsH")  # <- you asked to use MadsH
TABLE_ID   = os.getenv("TABLE_ID", "super_table")

SUPER_TABLE_FQN = f"`{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`"

# Core product columns to include (as per your “only include” list)
PRODUCT_COLS: list[str] = [
    "item_number",                      # <- COALESCE(pk_itemnumber, item_id) in your table build
    "item_description",
    "category",
    "category_code",
    "item_classification",
    "class_2",
    "class_3",
    "class_4",
    "class_4_id",
    "creation_date",
    "level0", "level1", "level2", "level3", "level4",
    "item_type",
    "countries_webshop_ind_active",
    "count_countries_webshop_active",
    "tag_webshop_active",
    "stop_purchase_ind",
    "brand_type",
    "brand_name",
    "brand_type_agg",
    "brand_type_best_better_good",
    "brand_type_ABC_brands",
    "brand_type_ta24",
    "turnover_eur_2021", "turnover_eur_2022", "turnover_eur_2023", "turnover_eur_2024",
    "orders_2021", "orders_2022", "orders_2023", "orders_2024",
    "tag_has_turnover",
    "avg_order_value_eur_2021", "avg_order_value_eur_2022", "avg_order_value_eur_2023", "avg_order_value_eur_2024",
    "quantity_sold_2021", "quantity_sold_2022", "quantity_sold_2023", "quantity_sold_2024",
    "avg_price_eur_2021", "avg_price_eur_2022", "avg_price_eur_2023", "avg_price_eur_2024",
    "list_price_turnover_eur_2021", "list_price_turnover_eur_2022", "list_price_turnover_eur_2023", "list_price_turnover_eur_2024",
    "margin_gross_eur_2021", "margin_gross_eur_2022", "margin_gross_eur_2023", "margin_gross_eur_2024",
    "margin_gross_perc_2021", "margin_gross_perc_2022", "margin_gross_perc_2023", "margin_gross_perc_2024",
    "customers_total_2021", "customers_total_2022", "customers_total_2023", "customers_total_2024",
    "sold_to_customers_new_2021", "sold_to_customers_new_2022", "sold_to_customers_new_2023", "sold_to_customers_new_2024",
    "customers_repeat_2021", "customers_repeat_2022", "customers_repeat_2023", "customers_repeat_2024",
    "customer_ordered_multiple_times_2021", "customer_ordered_multiple_times_2022", "customer_ordered_multiple_times_2023",
    "stock_on_hand_units", "stock_on_hand_eur", "warehouse_with_stock", "count_warehouse_with_stock",
    "supplier_count_id",
    "supplier_name_string",                 # <- used for Group Supplier filtering
    "supplier_code_string_best_abc",
    "supplier_name_string_best_abc",
    "supplier_relation_best_abc",
    "supplier_kraljic_matrix_best_abc",
    "supplier_abc_class_best_abc",
    "supplier_euro_w_mix_best_abc",
    "supplier_categories_bought_best_abc",
    "purchase_amount_eur_2020", "purchase_amount_eur_2021", "purchase_amount_eur_2022", "purchase_amount_eur_2023", "purchase_amount_eur_2024",
    "abc_costtotal_2021", "abc_costdistributiontotal_2021", "abc_costsalestotal_2021", "abc_costtechnologytotal_2021",
    "abc_profitability_2021", "abc_profitabily_percentage_2021", "abc_turnover_2021", "abc_costofgoodssold_2021",
    "abc_costtotal_2022", "abc_costdistributiontotal_2022", "abc_costsalestotal_2022", "abc_costtechnologytotal_2022",
    "abc_profitability_2022", "abc_profitabily_percentage_2022", "abc_turnover_2022", "abc_costofgoodssold_2022",
    "abc_costtotal_2023", "abc_costdistributiontotal_2023", "abc_costsalestotal_2023", "abc_costtechnologytotal_2023",
    "abc_profitability_2023", "abc_profitabily_percentage_2023", "abc_turnover_2023", "abc_costofgoodssold_2023",
    "abc_costtotal_2024", "abc_costdistributiontotal_2024", "abc_costsalestotal_2024", "abc_costtechnologytotal_2024",
    "abc_profitability_2024", "abc_profitabily_percentage_2024", "abc_turnover_2024", "abc_costofgoodssold_2024",
    "step_product_key", "step_product_id", "product_is_active",
    "STEP_CountryOfOrigin",                 # <- added from CMQ__ItemSupplierData__Combined_v5
]

# Attribute columns that exist row-wise in super_table (joined from Ciske)
ATTRIBUTE_COLS: list[str] = [
    "attribute_id",
    "attribute_description",
    "attribute_data_type",
    "attribute_value_id",
    "is_synthetic",
    "locale_independent_values",
    "localized_values",
    "unit_of_measurement_description",
    "unit_of_measurement_symbol",
    "attribute_sort_order",
    "attribute_value_sort_order",
]

DEFAULT_COLUMNS = PRODUCT_COLS + ATTRIBUTE_COLS

# -----------------------------------------------------------------------------
# Yearly column groupings & helper constant sets (cross-script alignment)
# -----------------------------------------------------------------------------
# The super_table table exposes purchase_amount_eur_YYYY (2020..2024 currently) and
# quantity_sold_YYYY (2021..2024). Separate purchase vs sales quantity naming used in
# purchase & supplier scripts (purchase_quantity_YYYY). We keep flexible groups below.

PURCHASE_AMOUNT_YEAR_COLS: list[str] = [
    # Dynamic growth: include 2020..2025 if present. These appear in pivoted purchase/supplier tables.
    "purchase_amount_eur_2020",
    "purchase_amount_eur_2021",
    "purchase_amount_eur_2022",
    "purchase_amount_eur_2023",
    "purchase_amount_eur_2024",
    "purchase_amount_eur_2025",
]

SALES_QUANTITY_YEAR_COLS: list[str] = [
    "quantity_sold_2021",
    "quantity_sold_2022",
    "quantity_sold_2023",
    "quantity_sold_2024",
]

PURCHASE_QUANTITY_YEAR_COLS: list[str] = [
    # Naming used in create_purchase_data.sql & create_supplier_data.sql (pivot outcome)
    "purchase_quantity_2020",
    "purchase_quantity_2021",
    "purchase_quantity_2022",
    "purchase_quantity_2023",
    "purchase_quantity_2024",
    "purchase_quantity_2025",
]

ORDERS_YEAR_COLS: list[str] = [
    "orders_2021", "orders_2022", "orders_2023", "orders_2024",
]

# Public lookups for external code (import from package __init__ already exports DEFAULT_COLUMNS)
YEAR_COLUMN_GROUPS: dict[str, list[str]] = {
    "purchase_amount": PURCHASE_AMOUNT_YEAR_COLS,
    "sales_quantity": SALES_QUANTITY_YEAR_COLS,
    "purchase_quantity": PURCHASE_QUANTITY_YEAR_COLS,
    "orders": ORDERS_YEAR_COLS,
}

# -----------------------------------------------------------------------------
# Small SQL helpers
# -----------------------------------------------------------------------------
def _sql_quote(vals: Iterable[str]) -> str:
    def _q(v: str) -> str:
        v = "" if v is None else str(v)
        return "'" + v.replace("'", "''") + "'"
    return ", ".join(_q(v) for v in vals)

def _in_clause(col: str, values: Optional[Iterable[str]]) -> Optional[str]:
    if not values:
        return None
    vals = list(values)
    if not vals:
        return None
    return f"{col} IN ({_sql_quote(vals)})"

def _like_any_clause(col: str, values: Optional[Iterable[str]]) -> Optional[str]:
    """Case-insensitive LIKE ANY helper (for keyword lists).

    Builds OR'ed LIKE clauses, skipping empty/blank keywords and properly escaping single quotes.
    Uses LOWER() on both sides for CI matching.
    """
    if not values:
        return None
    frags: list[str] = []
    for raw in values:
        v = (raw or "").strip()
        if not v:
            continue
        safe = v.replace("'", "''")  # basic quote escape
        frags.append(f"LOWER({col}) LIKE '%{safe.lower()}%'")
    return "(" + " OR ".join(frags) + ")" if frags else None

import re
def _regex_any_clause(col: str, values: Optional[Iterable[str]], *, whole_word: bool = True) -> Optional[str]:
    """Regex OR helper using BigQuery REGEXP_CONTAINS.

    Parameters
    ----------
    col : str
        Column name to search.
    values : iterable of str
        Keywords/phrases. Escaped for literal matching.
    whole_word : bool
        If True wrap each term with \b word boundaries.
    """
    if not values:
        return None
    parts: list[str] = []
    for raw in values:
        v = (raw or "").strip()
        if not v:
            continue
        esc = re.escape(v.lower())
        if whole_word:
            esc = r"\b" + esc + r"\b"
        parts.append(esc)
    if not parts:
        return None
    pattern = "|".join(parts)
    # (?i) inline for case-insensitive; keep original column (no LOWER needed)
    return f"REGEXP_CONTAINS({col}, r'(?i){pattern}')"

def _purchase_stop_clause(mode: Optional[str]) -> Optional[str]:
    """
    mode: 'strict' -> stop_purchase_ind = 'N'
          'lenient'-> stop_purchase_ind != 'Y' OR IS NULL
          None     -> no filter
    """
    if mode is None:
        return None
    m = mode.strip().lower()
    if m == "strict":
        return "stop_purchase_ind = 'N'"
    if m == "lenient":
        return "(stop_purchase_ind != 'Y' OR stop_purchase_ind IS NULL)"
    raise ValueError("purchase_stop must be one of: None, 'strict', 'lenient'")

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def fetch_super_table(
    *,
    columns: Optional[Sequence[str]] = None,
    class2: Optional[Sequence[str]] = None,
    class3: Optional[Sequence[str]] = None,
    class4: Optional[Sequence[str]] = None,
    brand_name: Optional[Sequence[str]] = None,
    country_of_origin: Optional[Sequence[str]] = None,   # STEP_CountryOfOrigin
    group_supplier: Optional[Sequence[str]] = None,      # supplier_name_string
    item_numbers: Optional[Sequence[str]] = None,        # item_number filter
    attribute_ids: Optional[Sequence[str]] = None,       # attribute_id filter
    keyword_in_desc: Optional[Sequence[str]] = None,     # keyword filter on item_description
    keyword_mode: str = "like",                         # 'like' | 'regex'
    regex_whole_word: bool = True,                      # applies if keyword_mode='regex'
    purchase_stop: Optional[str] = "strict",             # 'strict'|'lenient'|None
    limit: Optional[int] = None,
    debug_print_sql: bool = False,
) -> pd.DataFrame:
    """
    Pull rows from MadsH.super_table with flexible SQL-side filtering.

    Returns
    -------
    pd.DataFrame
    """
    cols = list(columns or DEFAULT_COLUMNS)
    select_cols = ",\n  ".join(cols)

    where_parts: list[str] = []

    # Exact IN filters
    if class2:            where_parts.append(_in_clause("class_2", class2))
    if class3:            where_parts.append(_in_clause("class_3", class3))
    if class4:            where_parts.append(_in_clause("class_4", class4))
    if brand_name:        where_parts.append(_in_clause("brand_name", brand_name))
    if country_of_origin: where_parts.append(_in_clause("STEP_CountryOfOrigin", country_of_origin))
    if group_supplier:    where_parts.append(_in_clause("supplier_name_string", group_supplier))
    if item_numbers:      where_parts.append(_in_clause("item_number", item_numbers))
    if attribute_ids:     where_parts.append(_in_clause("attribute_id", attribute_ids))

    # LIKE on item_description (keywords)
    if keyword_in_desc:
        km = (keyword_mode or "like").strip().lower()
        if km == "regex":
            kw_clause = _regex_any_clause("item_description", keyword_in_desc, whole_word=regex_whole_word)
        else:
            kw_clause = _like_any_clause("item_description", keyword_in_desc)
        if kw_clause:
            where_parts.append(kw_clause)

    # Purchase-stop policy
    ps = _purchase_stop_clause(purchase_stop)
    if ps:
        where_parts.append(ps)

    where_sql = ""
    if where_parts:
        where_sql = "WHERE " + "\n  AND ".join([w for w in where_parts if w])

        limit_sql = f"LIMIT {int(limit)}" if (isinstance(limit, int) and limit > 0) else ""

        query = f"""
        SELECT
            {select_cols}
        FROM {SUPER_TABLE_FQN}
        {where_sql}
        {limit_sql}
        """.strip()

    if debug_print_sql:
        print("=== SQL ===")
        print(query)

    bq = BigQueryConnector()
    df = bq.query(query)
    if df is None:
        return pd.DataFrame()

    # Optional type fixes (if you have product_utils in path)
    if coerce_dtypes:
        df = coerce_dtypes(df)
    return df


def fetch_super_table_for_clustering(
    *,
    class2: Optional[Sequence[str]] = None,
    class3: Optional[Sequence[str]] = None,
    class4: Optional[Sequence[str]] = None,
    brand_name: Optional[Sequence[str]] = None,
    purchase_stop: Optional[str] = "strict",
    limit: Optional[int] = None,
    debug_print_sql: bool = False,
) -> pd.DataFrame:
    """Minimal fetch tailored for clustering pipeline.

    Retrieves only the columns required for deriving the two clustering features
    (purchase amount total & quantity sold total) plus basic identifiers & filters.

    Returns
    -------
    pd.DataFrame
    """
    base_cols = [
        "item_number", "item_description", "class_2", "class_3", "class_4",
        "brand_name", "STEP_CountryOfOrigin", "supplier_name_string", "stop_purchase_ind",
    ]
    # Only include year columns that exist (avoid selecting non-existent future years)
    year_amount_cols = [c for c in PURCHASE_AMOUNT_YEAR_COLS if c in PRODUCT_COLS]
    year_qty_cols = [c for c in SALES_QUANTITY_YEAR_COLS if c in PRODUCT_COLS]
    cols = [c for c in base_cols + year_amount_cols + year_qty_cols if c in PRODUCT_COLS]
    return fetch_super_table(
        columns=cols,
        class2=class2,
        class3=class3,
        class4=class4,
        brand_name=brand_name,
        purchase_stop=purchase_stop,
        limit=limit,
        debug_print_sql=debug_print_sql,
    )


def pivot_attributes_wide(
    df: pd.DataFrame,
    *,
    index_cols: Sequence[str] = ("item_number",),
    value_col: str = "locale_independent_values",
    attr_col: str = "attribute_id",
    prefix: str = "attr_",
    first_value_only: bool = True,
) -> pd.DataFrame:
    """
    Convert attribute rows to wide columns keyed by attribute_id.

    If multiple rows per (item_number, attribute_id) exist, keeps the first by default.
    """
    needed = set(index_cols) | {value_col, attr_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for pivot: {sorted(missing)}")

    work = df.loc[:, list(needed)].copy()

    if first_value_only:
        # De-duplicate by (index..., attribute_id)
        work = (
            work
            .sort_values(list(index_cols) + [attr_col])
            .drop_duplicates(subset=list(index_cols) + [attr_col], keep="first")
        )

    wide = work.pivot(index=list(index_cols), columns=attr_col, values=value_col)
    wide = wide.add_prefix(prefix)
    wide = wide.reset_index()
    return wide


def compute_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate total columns if their components exist.

    Adds columns:
      - purchase_amount_eur_total
      - quantity_sold_total
      - purchase_quantity_total
      - orders_total

    Only sums across columns present in the DataFrame (safe on partial subsets).
    """
    work = df.copy()

    def _sum_cols(cols: list[str]) -> pd.Series:
        real = [c for c in cols if c in work.columns]
        if not real:
            return pd.Series([0] * len(work), index=work.index)
        return work[real].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)

    if any(c in work.columns for c in PURCHASE_AMOUNT_YEAR_COLS):
        work["purchase_amount_eur_total"] = _sum_cols(PURCHASE_AMOUNT_YEAR_COLS)
    if any(c in work.columns for c in SALES_QUANTITY_YEAR_COLS):
        work["quantity_sold_total"] = _sum_cols(SALES_QUANTITY_YEAR_COLS)
    if any(c in work.columns for c in PURCHASE_QUANTITY_YEAR_COLS):
        work["purchase_quantity_total"] = _sum_cols(PURCHASE_QUANTITY_YEAR_COLS)
    if any(c in work.columns for c in ORDERS_YEAR_COLS):
        work["orders_total"] = _sum_cols(ORDERS_YEAR_COLS)
    return work


def summarize_super_table(df: pd.DataFrame) -> dict:
    """Quick numeric summary of super_table style DataFrame.

    Returns a dictionary with sums of year columns & counts to assist diagnostics.
    """
    out = {}
    for group_name, cols in YEAR_COLUMN_GROUPS.items():
        present = [c for c in cols if c in df.columns]
        if present:
            # Sum numeric safely (2D: sum each column then total) for clarity
            year_sum = df[present].apply(pd.to_numeric, errors="coerce").fillna(0).sum().sum()
            out[f"{group_name}_year_cols_present"] = present
            out[f"{group_name}_year_total_sum"] = float(year_sum)
            out[f"{group_name}_nonzero_rows"] = int((df[present].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1) > 0).sum())
            out[f"{group_name}_max_single_year"] = float(df[present].apply(pd.to_numeric, errors="coerce").fillna(0).max().max())
    out["row_count"] = int(len(df))
    return out


def build_year_metrics_if_missing(
    df: pd.DataFrame,
    *,
    id_col: str = "item_number",
    year_col: str = "YearNumber",
    turnover_col: str = "TurnoverEuro",
    quantity_col: str = "QuantitySold",
    margin_col: str = "MarginEuro",
    list_price_turnover_col: str = "ListPriceTurnoverEuro",
    create_orders: bool = True,
) -> pd.DataFrame:
    """Derive wide year-suffixed metric columns from transactional rows when absent.

    If the super table was built from granular order lines (e.g., via `order_data`)
    but the aggregated columns like `turnover_eur_2023` / `quantity_sold_2024`
    are missing, this helper produces them. Existing year columns are left
    untouched.

    Parameters
    ----------
    df : DataFrame
        Transaction-level or partially aggregated data containing at least
        identifier and a year indicator.
    id_col : str, default 'item_number'
        Product identifier column to aggregate by. Adjust if your table uses
        `ProductId` instead.
    year_col : str, default 'YearNumber'
        Column with 4-digit year integers.
    turnover_col : str, default 'TurnoverEuro'
        Per-row turnover metric (sum basis).
    quantity_col : str, default 'QuantitySold'
        Per-row quantity metric (sum basis).
    margin_col : str, default 'MarginEuro'
        Per-row margin metric (sum basis) -> mapped to `margin_gross_eur_YEAR`.
    list_price_turnover_col : str, default 'ListPriceTurnoverEuro'
        Per-row list price turnover metric -> mapped to `list_price_turnover_eur_YEAR`.
    create_orders : bool, default True
        Whether to also create `orders_YEAR` columns counting distinct orders.

    Returns
    -------
    DataFrame with missing year-suffixed columns added.
    """
    work = df.copy()
    # Detect which year columns already exist to avoid recomputation
    existing_cols = set(work.columns)

    needed_map = {
        turnover_col: "turnover_eur_{year}",
        quantity_col: "quantity_sold_{year}",
        margin_col: "margin_gross_eur_{year}",
        list_price_turnover_col: "list_price_turnover_eur_{year}",
    }

    # Sanity: required raw columns present
    base_missing = [c for c in (id_col, year_col) if c not in existing_cols]
    if base_missing:
        # Nothing we can do; return original unchanged
        return work

    # Work only with needed raw metric columns that exist
    present_metric_cols = {raw: pat for raw, pat in needed_map.items() if raw in existing_cols}
    if not present_metric_cols and not create_orders:
        return work  # no metrics to build

    # Prepare numeric conversions
    for raw_col in present_metric_cols.keys():
        work[raw_col] = pd.to_numeric(work[raw_col], errors="coerce")
    if create_orders and "OrderNumber" in work.columns:
        # Keep as string for counting distinct
        work["OrderNumber"] = work["OrderNumber"].astype(str)

    # Aggregate by (id_col, year)
    group_keys = [id_col, year_col]
    agg_dict: dict[str, str] = {raw: "sum" for raw in present_metric_cols.keys()}
    if create_orders and "OrderNumber" in work.columns:
        agg_dict["OrderNumber"] = "nunique"

    grouped = work.groupby(group_keys, dropna=False).agg(agg_dict).reset_index()
    # Pivot each metric to wide year-suffixed columns
    for raw, pattern in present_metric_cols.items():
        wide = grouped.pivot(index=id_col, columns=year_col, values=raw)
        for yr, val in wide.items():  # pandas Series named by year
            col_name = pattern.format(year=int(yr))
            if col_name not in work.columns:
                # Map values back into main frame using id_col
                work = work.merge(wide[[yr]].rename(columns={yr: col_name}), left_on=id_col, right_index=True, how="left")
    if create_orders and "OrderNumber" in agg_dict:
        wide_o = grouped.pivot(index=id_col, columns=year_col, values="OrderNumber")
        for yr, val in wide_o.items():
            col_name = f"orders_{int(yr)}"
            if col_name not in work.columns:
                work = work.merge(wide_o[[yr]].rename(columns={yr: col_name}), left_on=id_col, right_index=True, how="left")

    # Fill NaNs from merges with 0 for numeric new columns
    new_cols = [c for c in work.columns if any(c.startswith(pref) for pref in ("turnover_eur_", "quantity_sold_", "margin_gross_eur_", "list_price_turnover_eur_", "orders_"))]
    for c in new_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0)

    return work


def fetch_distinct_values(
    column: str,
    *,
    where_sql: Optional[str] = None,
    limit: Optional[int] = None,
    debug_print_sql: bool = False,
) -> pd.DataFrame:
    """
    Useful for dropdowns: fetch distinct values from a single column.
    """
    safe_col = "".join(ch for ch in column if ch.isalnum() or ch in {"_",})
    if not safe_col:
        raise ValueError("Invalid column name.")

    where_clause = f"WHERE {where_sql}" if where_sql else ""
    limit_sql = f"LIMIT {int(limit)}" if (isinstance(limit, int) and limit > 0) else ""

    query = f"""
    SELECT DISTINCT {safe_col} AS value
    FROM {SUPER_TABLE_FQN}
    {where_clause}
    ORDER BY value
    {limit_sql}
    """.strip()

    if debug_print_sql:
        print("=== SQL ===")
        print(query)

    bq = BigQueryConnector()
    df = bq.query(query)
    return df if df is not None else pd.DataFrame()


# -----------------------------------------------------------------------------
# Convenience wrapper to mirror your notebook’s style (DataFrame-side filters)
# -----------------------------------------------------------------------------
def apply_df_filters(
    df: pd.DataFrame,
    *,
    class2: Optional[Sequence[str]] = None,
    class3: Optional[Sequence[str]] = None,
    class4: Optional[Sequence[str]] = None,
    brand_name: Optional[Sequence[str]] = None,
    country_of_origin: Optional[Sequence[str]] = None,
    group_supplier: Optional[Sequence[str]] = None,
    negate: bool = False,
    purchase_stop_mode: Optional[str] = None,  # 'strict'|'lenient'|None
) -> pd.DataFrame:
    """
    Apply the same filters client-side (useful when you've already fetched a chunk).
    Requires product_utils.* filter helpers to be importable.
    """
    out = df.copy()

    if class2 and filter_by_class2:                out = filter_by_class2(out, class2, negate=negate)
    if class3 and filter_by_class3:                out = filter_by_class3(out, class3, negate=negate)
    if class4 and filter_by_class4:                out = filter_by_class4(out, class4, negate=negate)
    if brand_name and filter_by_brand_name:        out = filter_by_brand_name(out, brand_name, negate=negate)
    if country_of_origin and filter_by_country_of_origin:
        out = filter_by_country_of_origin(out, country_of_origin, negate=negate)
    if group_supplier and filter_by_group_supplier:
        out = filter_by_group_supplier(out, group_supplier, negate=negate)

    if purchase_stop_mode:
        mode = purchase_stop_mode.strip().lower()
        if mode == "strict" and filter_not_purchase_stop:
            out = filter_not_purchase_stop(out, include_unknown=False)
        elif mode == "lenient" and filter_not_purchase_stop:
            out = filter_not_purchase_stop(out, include_unknown=True)

    return out


# -----------------------------------------------------------------------------
# Example quick usage (not executed on import)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Fetch a narrow sample with SQL-side filters
    df = fetch_super_table(
        columns=["item_number", "item_description", "class_3", "brand_name",
                 "STEP_CountryOfOrigin", "supplier_name_string", "stop_purchase_ind",
                 "attribute_id", "locale_independent_values"],
        class2=["Fasteners"],
        class3=["Threaded Rods"],
        brand_name=["Kramp"],
        country_of_origin=None,                  # e.g. ["Germany"]
        group_supplier=None,                     # e.g. ["Kerbl Group"]
        purchase_stop="strict",                  # 'strict'|'lenient'|None
        limit=10,
        debug_print_sql=True,
    )
    print(f"Fetched: {len(df):,} rows")

    # 2) Pivot attributes (optional)
    if not df.empty and "attribute_id" in df.columns:
        wide = pivot_attributes_wide(
            df,
            index_cols=("item_number", "item_description", "class_3", "brand_name"),
            value_col="locale_independent_values",
            attr_col="attribute_id",
            prefix="attr_",
        )
        print(f"Wide attrs shape: {wide.shape}")

    # 3) Totals & summary example
    df_tot = compute_totals(df)
    print("Summary:", summarize_super_table(df_tot))
