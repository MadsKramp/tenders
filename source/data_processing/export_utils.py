"""Export utilities (year-split PurchaseQuantity)."""
from __future__ import annotations

from typing import List, Optional
import os

import numpy as np
import pandas as pd

from .formatting_utils import safe_filename, write_excel_with_thousands


def export_to_excel(sheets: dict, output_path: str) -> None:
    """Write multiple DataFrames to an Excel file, one per sheet."""
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        for sheet_name, df in sheets.items():
            # Sheet names must be <= 31 chars and not contain special chars
            safe_sheet = str(sheet_name)[:31].replace(":", "_").replace("/", "_")
            df.to_excel(writer, sheet_name=safe_sheet, index=False)
    print(f"[export_to_excel] Wrote {len(sheets)} sheets to {output_path}")


__all__ = [
    "export_year_split_purchase_quantity",
    "fetch_year_purchase_quantity",
    "export_to_excel",
]

# ----------------------------------------------------------------------
# BigQuery query template for year-level PurchaseQuantity per product
# Uses RAW column names from the BigQuery table and aliases to:
#   ProductNumber, ProductDescription, Class3, Year, PurchaseQuantity
# ----------------------------------------------------------------------
YEAR_QTY_QUERY_TEMPLATE = """
SELECT
    ProductNumber,
    ANY_VALUE(ProductDescription) AS ProductDescription,
    class3 AS Class3,
    CASE
        WHEN REGEXP_CONTAINS(CAST(year_authorization AS STRING), r'^[0-9]{{4}}$')
            THEN CAST(year_authorization AS INT64)
        WHEN SAFE.PARSE_DATE('%Y-%m-%d', CAST(year_authorization AS STRING)) IS NOT NULL
            THEN EXTRACT(YEAR FROM SAFE.PARSE_DATE('%Y-%m-%d', CAST(year_authorization AS STRING)))
        ELSE CAST(SUBSTR(CAST(year_authorization AS STRING), 1, 4) AS INT64)
    END AS Year,
    SUM(purchase_quantity) AS PurchaseQuantity
FROM `{table}`
{where_clause}
GROUP BY ProductNumber, Class3, Year
ORDER BY ProductNumber, Year
"""


def fetch_year_purchase_quantity(
    bq,
    table: str = "kramp-sharedmasterdata-prd.MadsH.purchase_data",
) -> pd.DataFrame:
    """
    Fetch year-level purchase quantity per ProductNumber from BigQuery.

    Returns columns:
        ProductNumber, ProductDescription, Class3, Year, PurchaseQuantity
    """
    where_clause = ""
    query = YEAR_QTY_QUERY_TEMPLATE.format(table=table, where_clause=where_clause)
    try:
        df_year = bq.query(query)
    except Exception as e:
        print("Year-level purchase quantity query failed:", e)
        return pd.DataFrame()
    return df_year


def export_year_split_purchase_quantity(
    bq,
    output_dir: str,
    table: str = "kramp-sharedmasterdata-prd.MadsH.purchase_data",
    fmt_thousands: bool = True,
    merged_header_label: str = "PurchaseQuantity",
    segmentation_df: Optional[pd.DataFrame] = None,
    segmentation_col: str = "Segmentation",
) -> List[str]:
    """
    Export per-Class3 Excel files with year-split PurchaseQuantity.

    Logic:
      1. Fetch year-level quantities per ProductNumber from BigQuery.
      2. Pivot to wide format: one row per ProductNumber, columns PurchaseQuantity.YYYY.
      3. Merge enrichment (Segmentation + tech fields) from segmentation_df on ProductNumber.
      4. Merge Class3 from BigQuery (one Class3 per ProductNumber).
      5. Export one Excel per Class3 with:

         ProductNumber, ProductDescription, Segmentation, SalesRounding,
         Year Authorization, Din Standard, Iso Standard, Quality, Material,
         Surface Treatment, Head Shape, Thread Type, Head Height,
         Head Outside Diameter Width, Thread Diameter, Length, Height,
         Total Height, Width, Inside Diameter, Outside Diameter, Thickness,
         Designed For Thread, Total Length, Head Type, Thread Length,
         PurchaseQuantity.2021, PurchaseQuantity.2022, ...

    segmentation_df is expected to have PRETTY column names created during preprocessing.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Fetch year-level data
    df_year = fetch_year_purchase_quantity(bq, table=table)
    if df_year is None:
        print("Query returned None.")
        return []
    if df_year.empty:
        print("No year-level purchase quantity data; nothing exported.")
        return []

    # Ensure Year is numeric
    df_year["Year"] = pd.to_numeric(df_year["Year"], errors="coerce").astype("Int64")

    # 2) Pivot: one row per ProductNumber, year columns = PurchaseQuantity.YYYY
    #    Do NOT include any enrichment fields in the pivot index.
    pivot = (
        df_year.pivot_table(
            index=["ProductNumber"],
            columns="Year",
            values="PurchaseQuantity",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    pivot.columns.name = None

    # Rename year columns to PurchaseQuantity.YYYY
    year_cols = [c for c in pivot.columns if isinstance(c, (int, np.integer))]
    year_cols = sorted(year_cols)
    rename_years = {c: f"{merged_header_label}.{int(c)}" for c in year_cols}
    pivot = pivot.rename(columns=rename_years)
    year_cols_renamed = list(rename_years.values())

    # 3) Build Class3 map: one Class3 per ProductNumber from df_year
    class3_map = (
        df_year[["ProductNumber", "Class3"]]
        .drop_duplicates(subset=["ProductNumber"])
    )

    # 4) Merge enrichment from segmentation_df
    # segmentation_df must come from your preprocessed df and contain pretty names
    if segmentation_df is not None and "ProductNumber" in segmentation_df.columns:
        seg = segmentation_df.copy()
        seg["ProductNumber"] = seg["ProductNumber"].astype(str).str.strip()
        pivot["ProductNumber"] = pivot["ProductNumber"].astype(str).str.strip()

        merged = pivot.merge(seg, on="ProductNumber", how="left")
    else:
        merged = pivot.copy()

    # 5) Merge Class3 onto merged
    class3_map["ProductNumber"] = class3_map["ProductNumber"].astype(str).str.strip()
    merged["ProductNumber"] = merged["ProductNumber"].astype(str).str.strip()
    merged = merged.merge(class3_map, on="ProductNumber", how="left")

    # 6) Define the static columns to export (pretty names)
    static_cols = [
        "ProductNumber",
        "ProductDescription",
        segmentation_col,
        "SalesRounding",
        "Din Standard",
        "Iso Standard",
        "Quality",
        "Material",
        "Surface Treatment",
        "Head Shape",
        "Thread Type",
        "Head Height",
        "Head Outside Diameter Width",
        "Thread Diameter",
        "Length",
        "Height",
        "Total Height",
        "Width",
        "Inside Diameter",
        "Outside Diameter",
        "Thickness",
        "Designed For Thread",
        "Total Length",
        "Head Type",
        "Thread Length",
    ]

    # 7) Export per Class3
    written: List[str] = []

    if "Class3" not in merged.columns:
        print("No Class3 column after merge; nothing exported.")
        return []

    for c3, group in merged.groupby("Class3"):
        # Keep only static columns that exist + year columns
        export_cols = [c for c in static_cols if c in group.columns] + year_cols_renamed
        export_cols = [c for c in export_cols if c in group.columns]

        sub = group[export_cols].copy()

        # Clean year columns
        for yc in year_cols_renamed:
            if yc in sub.columns:
                sub[yc] = (
                    pd.to_numeric(sub[yc], errors="coerce")
                    .fillna(0)
                    .round(0)
                    .astype(int)
                )

        # Deduplicate per ProductNumber
        if "ProductNumber" in sub.columns:
            sub = sub.drop_duplicates(subset=["ProductNumber"], keep="first").copy()

        # Build a safe filename
        fname = safe_filename(f"ABC_Segmentation_{c3}_years") + ".xlsx"
        path = os.path.join(output_dir, fname)

        # Write Excel
        try:
            if fmt_thousands:
                write_excel_with_thousands(sub, path, thousand_cols=year_cols_renamed)
            else:
                export_to_excel({"Sheet1": sub}, path)
            written.append(path)
        except Exception as e:
            print(f"Export failed with formatting for {path}: {e}. Fallback plain Excel.")
            sub.to_excel(path, index=False)
            written.append(path)

        print(
            f"✅ Exported {path} "
            f"({len(sub)} rows) - year columns: {', '.join(year_cols_renamed)}"
        )

    return written
