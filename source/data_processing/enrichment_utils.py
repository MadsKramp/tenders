"""Utilities to enrich exported product files with additional product details."""
from __future__ import annotations
from typing import Iterable, Optional, List, Dict
import os
import inspect
import pandas as pd

__all__ = [
    "discover_product_details_function",
    "fetch_product_details_wide",
    "enrich_exports",
]

CANDIDATE_FUNC_NAMES = [
    "get_product_details_mapping",
    "fetch_product_details",
    "load_product_details",
    "get_product_details",
    "retrieve_product_details",
    "enrich_with_product_details",
    "enrich_products",
    "load_product_master",
    "fetch_product_master",
]


def discover_product_details_function(module) -> Optional[callable]:
    """Find the first callable in module matching expected candidate names.
    Falls back to scanning for any name containing 'product'."""
    for name in CANDIDATE_FUNC_NAMES:
        func = getattr(module, name, None)
        if callable(func):
            return func
    for name in dir(module):
        if "product" in name.lower():
            func = getattr(module, name)
            if callable(func):
                return func
    return None


def _resolve_parameters(detail_func, bq, product_numbers: List[str]):
    sig = inspect.signature(detail_func)
    connector_aliases = {"connector", "conn", "bq", "bq_connector", "client", "source_connector", "bqclient", "bq_client"}
    product_aliases = {"product_numbers", "products", "product_ids", "ids", "items"}
    args, kwargs = [], {}
    provided_connector = False
    provided_numbers = False
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    def _is_connector_name(name: str) -> bool:
        lname = name.lower()
        return lname in connector_aliases or any(x in lname for x in ["bq", "connector", "client"])

    def _is_products_name(name: str) -> bool:
        lname = name.lower()
        return lname in product_aliases or any(x in lname for x in ["product", "items", "ids"])

    for param in sig.parameters.values():
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            if param.default is inspect._empty:
                if _is_connector_name(param.name) and not provided_connector:
                    args.append(bq)
                    provided_connector = True
                elif _is_products_name(param.name) and not provided_numbers:
                    args.append(product_numbers)
                    provided_numbers = True
                else:
                    ann = param.annotation
                    if ann is not inspect._empty:
                        try:
                            from source.db_connect.bigquery_connector import BigQueryConnector as _BQC
                            if ann is _BQC and not provided_connector:
                                args.append(bq)
                                provided_connector = True
                                continue
                        except Exception:
                            pass
                    raise TypeError(f"Unsupported required parameter '{param.name}' in {detail_func.__name__}.")
            else:  # optional positional/keyword
                if _is_connector_name(param.name) and not provided_connector:
                    kwargs[param.name] = bq
                    provided_connector = True
                elif _is_products_name(param.name) and not provided_numbers:
                    kwargs[param.name] = product_numbers
                    provided_numbers = True
        elif param.kind == inspect.Parameter.KEYWORD_ONLY and param.default is inspect._empty:
            if _is_connector_name(param.name) and not provided_connector:
                kwargs[param.name] = bq
                provided_connector = True
            elif _is_products_name(param.name) and not provided_numbers:
                kwargs[param.name] = product_numbers
                provided_numbers = True

    if not provided_numbers:
        if any(_is_products_name(p.name) for p in sig.parameters.values()):
            target = next(p.name for p in sig.parameters.values() if _is_products_name(p.name))
            kwargs[target] = product_numbers
            provided_numbers = True
        elif accepts_kwargs:
            kwargs["product_numbers"] = product_numbers
            provided_numbers = True
    if not provided_numbers:
        raise TypeError(f"{detail_func.__name__} must accept product numbers as an argument.")

    if not provided_connector:
        for p in sig.parameters.values():
            if _is_connector_name(p.name):
                kwargs[p.name] = bq
                provided_connector = True
                break
    if not provided_connector:
        for p in sig.parameters.values():
            ann = p.annotation
            if ann is not inspect._empty:
                try:
                    from source.db_connect.bigquery_connector import BigQueryConnector as _BQC
                    if ann is _BQC:
                        kwargs[p.name] = bq
                        provided_connector = True
                        break
                except Exception:
                    pass
    return args, kwargs


def fetch_product_details_wide(bq, product_numbers: Iterable[str], module) -> pd.DataFrame:
    """Fetch product details and return a wide DataFrame keyed by ProductNumber."""
    product_numbers = sorted(set(str(p).strip() for p in product_numbers if str(p).strip()))
    if not product_numbers:
        return pd.DataFrame(columns=["ProductNumber"])

    detail_func = discover_product_details_function(module)
    if detail_func is None:
        raise AttributeError("Could not locate a product detail fetcher in supplied module.")

    args, kwargs = _resolve_parameters(detail_func, bq, product_numbers)
    detail_df = detail_func(*args, **kwargs)
    if detail_df is None:
        raise ValueError(f"{detail_func.__name__} returned None.")
    if not isinstance(detail_df, pd.DataFrame):
        detail_df = pd.DataFrame(detail_df)
    if detail_df.empty:
        return pd.DataFrame(columns=["ProductNumber"])

    details = detail_df.copy()
    if details.columns.duplicated().any():
        details = details.loc[:, ~details.columns.duplicated()]
    renames = {}
    for col in details.columns:
        low = col.lower()
        if low in {"productnumber", "product_number", "product_id", "productid"}:
            renames[col] = "ProductNumber"
    if renames:
        details = details.rename(columns=renames)
    if "ProductNumber" not in details.columns:
        raise KeyError("Result must include 'ProductNumber'.")
    details["ProductNumber"] = details["ProductNumber"].astype(str).str.strip().replace({"nan": "", "None": ""})
    details = details[details["ProductNumber"] != ""]
    if details.empty:
        return pd.DataFrame(columns=["ProductNumber"])

    # Long-form pivot if present
    if {"detail_name", "detail_value"} <= set(details.columns):
        details = details.pivot_table(
            index="ProductNumber",
            columns="detail_name",
            values="detail_value",
            aggfunc="first"
        ).reset_index()
        details.columns.name = None
        details.columns = [str(c) for c in details.columns]
    return details


def enrich_exports(directory: str,
                   bq,
                   product_utils_module,
                   preserve_year_format: bool = True,
                   merged_header_label: str = "PurchaseQuantity",
                   output_suffix: str = "_enriched") -> Dict[str, int]:
    """Enrich all Excel files in directory with product details.

    Returns dict with summary: {"files_enriched": int, "new_columns_added_total": int}.
    Skips files lacking ProductNumber column.
    """
    if not os.path.isdir(directory):
        print(f"Directory not found: {directory}")
        return {"files_enriched": 0, "new_columns_added_total": 0}

    excel_paths = [
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.lower().endswith(".xlsx")
    ]
    if not excel_paths:
        print("No Excel files to enrich.")
        return {"files_enriched": 0, "new_columns_added_total": 0}

    exports_cache = {}
    product_numbers = set()
    for path in excel_paths:
        # Initial read: header row might be a merged top-level header (row 0) from year-split export.
        df = pd.read_excel(path, dtype={"ProductNumber": str})
        # Heuristic: if more than 2 columns and any column names after the first two are empty/unnamed,
        # re-read using header=1 to treat second row as the true header (year columns etc.).
        if df.shape[1] >= 3:
            unnamed_after_two = [c for c in df.columns[2:] if (not isinstance(c, str)) or c.startswith("Unnamed") or c.strip() == ""]
            if unnamed_after_two:
                try:
                    df = pd.read_excel(path, dtype={"ProductNumber": str}, header=1)
                except Exception:
                    pass
        exports_cache[path] = df
        if "ProductNumber" not in df.columns:
            continue
        series = df["ProductNumber"].astype(str).str.strip().replace({"nan": "", "None": ""})
        product_numbers.update(series[series != ""].tolist())

    if not product_numbers:
        print("No product numbers found; enrichment skipped.")
        return {"files_enriched": 0, "new_columns_added_total": 0}

    # Fetch details
    details = fetch_product_details_wide(bq, list(product_numbers), product_utils_module)
    if details.empty:
        sample = list(product_numbers)[:10]
        print(f"Product details fetch returned empty; inspected {len(product_numbers)} product numbers (sample: {sample}). Nothing to merge.")
        return {"files_enriched": 0, "new_columns_added_total": 0}

    detail_cols = [c for c in details.columns if c != "ProductNumber"]
    files_enriched = 0
    new_columns_total = 0

    for path, export_df in exports_cache.items():
        if "ProductNumber" not in export_df.columns:
            continue
        prod_series = export_df["ProductNumber"].astype(str).str.strip().replace({"nan": "", "None": ""})
        meta_mask = prod_series == ""  # preserve top metadata rows
        meta_rows = export_df.loc[meta_mask].copy()
        data_rows = export_df.loc[~meta_mask].copy()
        if data_rows.empty:
            continue
        data_rows["ProductNumber"] = prod_series[~meta_mask]
        merged = data_rows.merge(details, on="ProductNumber", how="left", suffixes=("", "_detail"))
        for col in detail_cols:
            detail_col = f"{col}_detail"
            if col in merged.columns and detail_col in merged.columns:
                merged[col] = merged[col].where(merged[col].notna(), merged[detail_col])
                merged.drop(columns=detail_col, inplace=True)
            elif detail_col in merged.columns:
                merged.rename(columns={detail_col: col}, inplace=True)

        original_cols = list(export_df.columns)
        new_cols = [c for c in detail_cols if c not in original_cols]
        final_cols = original_cols + new_cols
        merged = merged.reindex(columns=final_cols)
        if not meta_rows.empty:
            meta_rows = meta_rows.reindex(columns=final_cols)
            out_df = pd.concat([meta_rows, merged], ignore_index=True)
        else:
            out_df = merged

        # Detect year-split format (headers with years as separate columns, possibly merged header previously)
        import re
        # Detect year columns either plain '2021' or combined label 'PurchaseQuantity.2021'
        year_cols = []
        for c in out_df.columns:
            if c.isdigit():
                year_cols.append(c)
            elif re.match(rf'^{merged_header_label}\.\d{{4}}$', c):
                year_cols.append(c)
        out_path = path  # default overwrite
        if output_suffix:
            root, ext = os.path.splitext(path)
            out_path = f"{root}{output_suffix}{ext}"
        if preserve_year_format and year_cols:
            try:
                with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
                    out_df.to_excel(writer, index=False)
                    workbook = writer.book
                    worksheet = writer.sheets["Sheet1"]
                    qty_fmt = workbook.add_format({"num_format": "#,##0"})
                    for col_idx, col_name in enumerate(out_df.columns):
                        if col_name in year_cols:
                            worksheet.set_column(col_idx, col_idx, 12, qty_fmt)
            except Exception as e:
                print(f"Formatting preservation failed for {out_path}: {e}. Writing plain file.")
                out_df.to_excel(out_path, index=False)
        else:
            out_df.to_excel(out_path, index=False)
        files_enriched += 1
        new_columns_total += len(new_cols)
        print(f"✅ Enriched {os.path.basename(out_path)} (+{len(new_cols)} cols).")

    if files_enriched == 0:
        print("No files updated; no matching product numbers found.")
    else:
        print(f"Finished enrichment: {files_enriched} files updated; {new_columns_total} new columns added.")
    return {"files_enriched": files_enriched, "new_columns_added_total": new_columns_total}
