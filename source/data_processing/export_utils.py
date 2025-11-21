"""Export utilities (year-split PurchaseQuantity)."""
from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import pandas as pd

from .formatting_utils import safe_filename, write_excel_with_thousands

__all__ = ["export_year_split_purchase_quantity", "fetch_year_purchase_quantity"]

YEAR_QTY_QUERY_TEMPLATE = """
SELECT
    ProductNumber,
    ANY_VALUE(ProductDescription) AS ProductDescription,  -- avoid duplicate rows when description varies by year
    class3 AS Class3,
    ANY_VALUE(salesRounding) AS salesRounding,
    -- Robust year derivation: numeric year_authorization stays; try date parse; fallback to first 4 chars
    CASE
        WHEN REGEXP_CONTAINS(CAST(year_authorization AS STRING), r'^[0-9]{{4}}$') THEN CAST(year_authorization AS INT64)
        WHEN SAFE.PARSE_DATE('%Y-%m-%d', CAST(year_authorization AS STRING)) IS NOT NULL THEN EXTRACT(YEAR FROM SAFE.PARSE_DATE('%Y-%m-%d', CAST(year_authorization AS STRING)))
        ELSE CAST(SUBSTR(CAST(year_authorization AS STRING),1,4) AS INT64)
    END AS Year,
    SUM(purchase_quantity) AS PurchaseQuantity
FROM `{table}`
{where_clause}
GROUP BY ProductNumber, Class3, Year
ORDER BY ProductNumber, Year
"""

def _bq_array_literal(values: list[str]) -> str:
    def q(s: str) -> str:
        return "'" + s.replace("'","''") + "'"
    return "[" + ",".join(q(v) for v in values) + "]"

def fetch_year_purchase_quantity(bq,
                                 table: str = "kramp-sharedmasterdata-prd.MadsH.purchase_data") -> pd.DataFrame:
    """Fetch raw year-level purchase quantity aggregated per product."""
    where_clause = ""
    query = YEAR_QTY_QUERY_TEMPLATE.format(table=table, where_clause=where_clause)
    try:
        df_year = bq.query(query)
    except Exception as e:
        print("Year-level purchase quantity query failed:", e)
        return pd.DataFrame()
    return df_year

def export_year_split_purchase_quantity(bq,
                                            # Remove duplicate ProductNumber columns after merge (keep first)
                                        output_dir: str,
                                        table: str = "kramp-sharedmasterdata-prd.MadsH.purchase_data",
                                        fmt_thousands: bool = True,
                                        merged_header_label: str = "PurchaseQuantity",
                                        segmentation_df: Optional[pd.DataFrame] = None,
                                        segmentation_col: str = "abc_tier") -> List[str]:
    """
    Query purchase data and export per-Class3 Excel files with year columns.

    Returns list of file paths written. PurchaseQuantity columns are formatted with
    thousand separators if fmt_thousands=True.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Fetch data
    df_year = fetch_year_purchase_quantity(bq, table=table)
    if df_year is None:
        print("Query returned None.")
        return []

    if df_year.empty:
        print("No year-level purchase quantity data; nothing exported.")
        return []

    df_year["Year"] = pd.to_numeric(df_year["Year"], errors="coerce").astype("Int64")
    REQUIRED_EXPORT_FIELDS = [
        'year_authorization', 'ProductNumber', 'ProductDescription', 'purchase_amount_eur',
        'head_shape', 'thread_type', 'head_height', 'head_outside_diameter_width', 'quality',
        'surface_treatment', 'material', 'din_standard', 'weight_per_100_pcs', 'content_in_sales_unit',
        'thread_diameter', 'length', 'height', 'total_height', 'width', 'iso_standard', 'inside_diameter',
        'outside_diameter', 'thickness', 'designed_for_thread', 'total_length', 'head_type', 'thread_length',
        'salesRounding'
    ]
    # Merge required/enrichment/segmentation fields BEFORE pivot
    if segmentation_df is not None and "ProductNumber" in segmentation_df.columns:
        # Prepare mapping for all required fields
        required_fields_present = [f for f in REQUIRED_EXPORT_FIELDS if f in segmentation_df.columns and f != "ProductNumber"]
        if required_fields_present or "ProductNumber" in segmentation_df.columns:
            # Ensure ProductNumber exists and is string
            if "ProductNumber" not in segmentation_df.columns:
                raise KeyError("ProductNumber column missing in segmentation_df.")
            required_map = segmentation_df[["ProductNumber"] + required_fields_present].drop_duplicates()
            required_map["ProductNumber"] = required_map["ProductNumber"].apply(lambda x: str(x).strip() if pd.notnull(x) else "")
            df_year["ProductNumber"] = df_year["ProductNumber"].apply(lambda x: str(x).strip() if pd.notnull(x) else "")
            df_year = df_year.merge(required_map, on="ProductNumber", how="left")
        # Prepare segmentation mapping
        if segmentation_col and segmentation_col in segmentation_df.columns:
            seg_map = segmentation_df[["ProductNumber", segmentation_col]].drop_duplicates()
            seg_map["ProductNumber"] = seg_map["ProductNumber"].astype(str).str.strip()
            if segmentation_col not in df_year.columns:
                df_year = df_year.merge(seg_map, on="ProductNumber", how="left")

    # Remove duplicate ProductNumber columns after merge (keep first)
    prod_cols = [col for col in df_year.columns if col.startswith("ProductNumber")]
    if len(prod_cols) > 1:
        keep_col = "ProductNumber" if "ProductNumber" in prod_cols else prod_cols[0]
        for col in prod_cols:
            if col != keep_col:
                df_year = df_year.drop(columns=[col])

    # Now define all variables for pivot and export
    year_cols = None
    rename_map = None
    year_cols_str = None
    combined_map = None
    combined_year_cols = None
    written = []


    # Only one clean implementation below
    pivot = (
        df_year.pivot_table(
            index=["Class3", "ProductNumber"] + [f for f in REQUIRED_EXPORT_FIELDS if f in df_year.columns and f != "ProductNumber"] + ([segmentation_col] if segmentation_col and segmentation_col in df_year.columns else []),
            columns="Year",
            values="PurchaseQuantity",
            aggfunc="sum",
            fill_value=0,
        ).reset_index()
    )
    pivot.columns.name = None
    pivot = pivot.drop_duplicates(subset=["ProductNumber"], keep="first").copy()

    year_cols = [c for c in pivot.columns if isinstance(c, (int, np.integer))]
    year_cols = sorted(year_cols)
    rename_map = {c: str(c) for c in year_cols}
    pivot = pivot.rename(columns=rename_map)
    year_cols_str = [rename_map[c] for c in year_cols]
    combined_map = {yc: f"{merged_header_label}.{yc}" for yc in year_cols_str}
    pivot = pivot.rename(columns=combined_map)
    combined_year_cols = [combined_map[yc] for yc in year_cols_str]

    written: List[str] = []
    for c3, group in pivot.groupby("Class3"):
        base_cols = [f for f in REQUIRED_EXPORT_FIELDS if f in group.columns]
        if segmentation_col and segmentation_col in group.columns:
            base_cols.append(segmentation_col)
        export_cols = base_cols + combined_year_cols
        export_cols = [col for col in export_cols if col in group.columns]
        sub = group[export_cols].copy()
        for yc in combined_year_cols:
            if yc in sub.columns:
                sub[yc] = pd.to_numeric(sub[yc], errors="coerce").fillna(0).round(0).astype(int)
        sub = sub.drop_duplicates(subset=["ProductNumber"], keep="first").copy()
        fname = safe_filename(f"ABC_Segmentation_{c3}_years") + ".xlsx"
        path = os.path.join(output_dir, fname)
        try:
            try:
                import xlsxwriter  # noqa: F401
            except ModuleNotFoundError:
                import sys, subprocess
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "xlsxwriter"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    import xlsxwriter  # noqa: F401
                except Exception:
                    raise
            with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
                sub.to_excel(writer, index=False)
                workbook = writer.book
                worksheet = writer.sheets["Sheet1"]
                qty_fmt = workbook.add_format({"num_format": "#,##0"}) if fmt_thousands else None
                for col_idx, col_name in enumerate(sub.columns):
                    if col_name in combined_year_cols and qty_fmt is not None:
                        worksheet.set_column(col_idx, col_idx, 12, qty_fmt)
                written.append(path)
        except Exception as e:
            print(f"xlsxwriter header layering failed for {path}: {e}. Fallback plain export.")
            if fmt_thousands:
                write_excel_with_thousands(sub, path, thousand_cols=combined_year_cols)
            else:
                sub.to_excel(path, index=False)
            written.append(path)
        print(f"✅ Exported {path} ({len(sub)} rows) - year columns: {', '.join(combined_year_cols)}")


