"""
tender_material_utils.py — Utilities for the MadsH.tender_material table (EU)

- BigQuery fetches from: kramp-sharedmasterdata-prd.MadsH.tender_material
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
    from product_utils import (
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
TABLE_ID   = os.getenv("TABLE_ID", "tender_material")

TENDER_MATERIAL_FQN = f"`{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`"

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

# Attribute columns that exist row-wise in tender_material (joined from Ciske)
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
    """Case-insensitive LIKE ANY helper (for keyword lists)."""
    if not values:
        return None
    frags = [f"LOWER({col}) LIKE '%' || LOWER({_sql_quote([v])[0]}) || '%'" for v in values]
    return "(" + " OR ".join(frags) + ")" if frags else None

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
def fetch_tender_material(
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
    keyword_in_desc: Optional[Sequence[str]] = None,     # LIKE filter on item_description
    purchase_stop: Optional[str] = "strict",             # 'strict'|'lenient'|None
    limit: Optional[int] = None,
    debug_print_sql: bool = False,
) -> pd.DataFrame:
    """
    Pull rows from MadsH.tender_material with flexible SQL-side filtering.

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
        like_clause = _like_any_clause("item_description", keyword_in_desc)
        if like_clause:
            where_parts.append(like_clause)

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
    FROM {TENDER_MATERIAL_FQN}
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
    FROM {TENDER_MATERIAL_FQN}
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
    df = fetch_tender_material(
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
