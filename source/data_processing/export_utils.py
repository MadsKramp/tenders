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
                                 table: str = "kramp-sharedmasterdata-prd.MadsH.purchase_data",
                                 level2_filter: Optional[list[str]] = None,
                                 level2_column_candidates: tuple[str, ...] = ("class2","Class2","level2","Level2")) -> pd.DataFrame:
    """Fetch raw year-level purchase quantity aggregated per product with optional Level2 filter.

    level2_filter: list of Level2/category names (e.g. ['Threaded Fasteners']). If provided we attempt to
    locate a Level2 column (from candidates) in the source table via probe. If found, add WHERE clause
    restricting rows to those Level2 values.
    """
    where_clause = ""
    if level2_filter:
        # Probe table for available columns to pick correct Level2 column name
        try:
            probe_df = bq.query(f"SELECT * FROM `{table}` LIMIT 1")
            if probe_df is not None and not probe_df.empty:
                cols = set(probe_df.columns)
                chosen = None
                for cand in level2_column_candidates:
                    if cand in cols:
                        chosen = cand
                        break
                if chosen:
                    # Build array literal
                    values = [str(v).strip() for v in level2_filter if str(v).strip()]
                    if values:
                        where_clause = f"WHERE {chosen} IN UNNEST({_bq_array_literal(values)})"
                else:
                    print("Level2 filter ignored: no candidate Level2 column present in table.")
            else:
                print("Level2 filter probe returned empty; skipping Level2 restriction.")
        except Exception as e:
            print("Level2 probe failed; proceeding without Level2 filter:", e)
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
                                        segmentation_col: str = "abc_tier",
                                        level2_filter: Optional[list[str]] = None) -> List[str]:
    """
    Query purchase data and export per-Class3 Excel files with year columns.

    Supports optional Level2 filtering: pass a list of Level2/category names (e.g. ['54 | Fasteners'])
    via the level2_filter argument to restrict exports to products matching those categories.
    The function will auto-detect the correct Level2 column in the source table.

    Returns list of file paths written. PurchaseQuantity columns are formatted with
    thousand separators if fmt_thousands=True.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Fetch data with optional Level2 filter
    df_year = fetch_year_purchase_quantity(bq, table=table, level2_filter=level2_filter)
    if df_year is None:
        print("Query returned None.")
        return []

    if df_year is None or df_year.empty:
        print("No year-level purchase quantity data; nothing exported.")
        return []

    df_year["Year"] = pd.to_numeric(df_year["Year"], errors="coerce").astype("Int64")
    pivot = (
        df_year.pivot_table(
            index=["Class3", "ProductNumber", "ProductDescription"],
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
    # Prepare segmentation mapping if provided
    seg_map = None
    if segmentation_df is not None and "ProductNumber" in segmentation_df.columns and segmentation_col in segmentation_df.columns:
        try:
            seg_map = (segmentation_df[["ProductNumber", segmentation_col]]
                       .drop_duplicates()
                       .assign(ProductNumber=lambda d: d["ProductNumber"].astype(str).str.strip()))
        except Exception:
            seg_map = None
    for c3, group in pivot.groupby("Class3"):
        sub = group[["ProductNumber", "ProductDescription"] + combined_year_cols].copy()
        if seg_map is not None:
            sub["ProductNumber"] = sub["ProductNumber"].astype(str).str.strip()
            sub = sub.merge(seg_map, on="ProductNumber", how="left")
            if segmentation_col in sub.columns:
                # Reorder columns to place segmentation before year columns
                sub = sub.reindex(columns=["ProductNumber", "ProductDescription", segmentation_col] + combined_year_cols)
        # enforce integers
        for yc in combined_year_cols:
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
