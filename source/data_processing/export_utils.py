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

    if df_year is None or df_year.empty:
        print("No year-level purchase quantity data; nothing exported.")
        return []

    df_year["Year"] = pd.to_numeric(df_year["Year"], errors="coerce").astype("Int64")
    pivot = (
        df_year.pivot_table(
            index=["Class3", "ProductNumber", "ProductDescription", "salesRounding"],
            columns="Year",
            values="PurchaseQuantity",
            aggfunc="sum",
            fill_value=0,
        ).reset_index()
    )
    pivot.columns.name = None
    # Deduplicate products within Class3 (keep first description)
    dup_mask = pivot.duplicated(subset=["Class3", "ProductNumber"], keep="first")
    if dup_mask.any():
        removed = int(dup_mask.sum())
        pivot = pivot[~dup_mask].copy()
        print(f"Removed {removed} duplicate product rows caused by varying ProductDescription.")

    year_cols = [c for c in pivot.columns if isinstance(c, (int, np.integer))]
    year_cols = sorted(year_cols)
    rename_map = {c: str(c) for c in year_cols}
    pivot = pivot.rename(columns=rename_map)
    year_cols_str = [rename_map[c] for c in year_cols]
    # Combine header label + year into single column names e.g. PurchaseQuantity.2021
    combined_map = {yc: f"{merged_header_label}.{yc}" for yc in year_cols_str}
    pivot = pivot.rename(columns=combined_map)
    combined_year_cols = [combined_map[yc] for yc in year_cols_str]

    written: List[str] = []
    # Prepare segmentation and enrichment mapping if provided
    seg_map = None
    enrichment_map = None
    ENRICHMENT_FIELDS = [
        'head_shape', 'thread_type', 'head_height', 'head_outside_diameter_width', 'quality',
        'surface_treatment', 'material', 'din_standard', 'weight_per_100_pcs', 'content_in_sales_unit',
        'thread_diameter', 'length', 'height', 'total_height', 'width', 'iso_standard', 'inside_diameter',
        'outside_diameter', 'thickness', 'designed_for_thread', 'total_length', 'head_type', 'thread_length'
    ]
    if segmentation_df is not None and "ProductNumber" in segmentation_df.columns:
        # Prepare segmentation mapping
        if segmentation_col and segmentation_col in segmentation_df.columns:
            try:
                seg_map = (segmentation_df[["ProductNumber", segmentation_col]]
                           .drop_duplicates()
                           .assign(ProductNumber=lambda d: d["ProductNumber"].astype(str).str.strip()))
            except Exception:
                seg_map = None
        # Prepare enrichment mapping
        enrichment_fields_present = [f for f in ENRICHMENT_FIELDS if f in segmentation_df.columns]
        if enrichment_fields_present:
            try:
                enrichment_map = (segmentation_df[["ProductNumber"] + enrichment_fields_present]
                                  .drop_duplicates()
                                  .assign(ProductNumber=lambda d: d["ProductNumber"].astype(str).str.strip()))
            except Exception:
                enrichment_map = None
    # Merge enrichment fields into pivot before export
    if enrichment_map is not None:
        pivot = pivot.merge(enrichment_map, on="ProductNumber", how="left")
    # Merge segmentation column if not present
    if seg_map is not None and segmentation_col and segmentation_col not in pivot.columns:
        pivot = pivot.merge(seg_map, on="ProductNumber", how="left")
    for c3, group in pivot.groupby("Class3"):
        # Start with required columns
        base_cols = ["ProductNumber", "ProductDescription", "salesRounding"]
        # Add segmentation if available and present in group
        if segmentation_col and segmentation_col in group.columns:
            base_cols.append(segmentation_col)
        # Add enrichment fields if present in group
        enrichment_present = [f for f in ENRICHMENT_FIELDS if f in group.columns]
        # Final columns: base + enrichment + year cols
        export_cols = base_cols + enrichment_present + combined_year_cols
        # Only include columns that exist in the group
        export_cols = [col for col in export_cols if col in group.columns]
        sub = group[export_cols].copy()
        # enforce integers for year columns
        for yc in combined_year_cols:
            if yc in sub.columns:
                sub[yc] = pd.to_numeric(sub[yc], errors="coerce").fillna(0).round(0).astype(int)
        fname = safe_filename(f"ABC_Segmentation_{c3}_years") + ".xlsx"
        path = os.path.join(output_dir, fname)

        # Write single header row with combined column names
        try:
            try:
                import xlsxwriter  # noqa: F401
            except ModuleNotFoundError:
                # Attempt lightweight install (non-fatal)
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
                # Apply number format to each combined year column
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

    if not written:
        print("No per-Class3 exports produced.")
    else:
        print("Finished year-split exports.")
    return written
