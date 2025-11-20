"""Utilities for normalizing and resolving core columns in purchase datasets.

Functions here encapsulate the column discovery / synthesis logic that previously lived
inline in the notebook. Each resolver returns the modified DataFrame (copy) and any
auxiliary outputs (like the resolved column name).
"""
from __future__ import annotations
from typing import Iterable, Tuple, Optional
import pandas as pd

__all__ = [
    "resolve_spend",
    "resolve_class3",
    "resolve_group_vendor",
    "resolve_product_identifiers",
    "resolve_all",
]


def _resolve_column(df: pd.DataFrame,
                    target_name: str,
                    candidates: Iterable[str],
                    synth_func=None,
                    warn_msg: Optional[str] = None) -> Tuple[pd.DataFrame, bool]:
    """Case-insensitive resolution of a column.

    If one of the candidate names (case-insensitive) exists, rename it to `target_name`.
    Else if `synth_func` is provided, synthesize the column.
    Returns (updated_df, found_bool).
    """
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            original = lower_map[key]
            if original != target_name:
                df = df.copy()
                df = df.rename(columns={original: target_name})
            return df, True
    # Synthesize
    if synth_func is not None and target_name not in df.columns:
        df = df.copy()
        df[target_name] = synth_func(df)
        if warn_msg:
            print(warn_msg)
        return df, True
    return df, target_name in df.columns


def resolve_spend(df: pd.DataFrame, spend_col: str = "total_spend") -> Tuple[pd.DataFrame, str]:
    """Resolve or derive the spend column.

    Priority:
    1. Existing spend_col
    2. Candidate direct numeric spend columns
    3. purchase_amount_eur * purchase_quantity
    4. Fallback to Amount Eur
    Raises ValueError if no path found.
    """
    if spend_col in df.columns:
        return df, spend_col

    candidates = ["total_spend", "purchase_amount_eur", "amount_eur", "Amount Eur", "total_purchase_amount_eur"]
    candidate = next((c for c in candidates if c in df.columns), None)
    if candidate:
        out = df.copy()
        out[spend_col] = pd.to_numeric(out[candidate], errors="coerce").fillna(0)
        return out, spend_col

    if {"purchase_amount_eur", "purchase_quantity"} <= set(df.columns):
        out = df.copy()
        out[spend_col] = (
            pd.to_numeric(out["purchase_amount_eur"], errors="coerce").fillna(0)
            * pd.to_numeric(out["purchase_quantity"], errors="coerce").fillna(0)
        )
        return out, spend_col

    if {"Amount Eur", "Quantity"} <= set(df.columns):
        out = df.copy()
        out[spend_col] = pd.to_numeric(out["Amount Eur"], errors="coerce").fillna(0)
        return out, spend_col

    raise ValueError(f"Unable to derive spend column. Columns: {df.columns.tolist()}")


def resolve_class3(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Resolve / construct the Class3 column.

    Handles multiple ambiguous 'Class' columns by renaming them into a simple hierarchy.
    Synthesizes Class3 from a nearby class level if needed.
    """
    if "Class3" in df.columns:
        return df, "Class3"

    class3_candidates = [
        "Class3", "class3", "class_3", "Class3Description", "class3description", "Class3Number", "class3number"
    ]
    existing = next((c for c in class3_candidates if c in df.columns), None)
    if existing:
        if existing != "Class3":
            df = df.rename(columns={existing: "Class3"})
        return df, "Class3"

    generic_class_cols = [c for c in df.columns if c.lower() == "class"]
    if generic_class_cols:
        out = df.copy()
        if len(generic_class_cols) > 1:
            new_cols = list(out.columns)
            seen = {}
            for i, col in enumerate(new_cols):
                if col.lower() == "class":
                    seen[col] = seen.get(col, 0) + 1
                    if seen[col] > 1:
                        new_cols[i] = f"Class_dup{seen[col]-1}"
            out.columns = new_cols
            generic_class_cols = [c for c in out.columns if c.lower() in {"class", "class_dup0", "class_dup1", "class_dup2"}]
        ordered = generic_class_cols
        mapping = {}
        if len(ordered) >= 1:
            mapping[ordered[0]] = "Class3"
        if len(ordered) >= 2:
            mapping[ordered[1]] = "Class4"
        out = out.rename(columns=mapping)
        if "Class3" not in out.columns:
            raise KeyError(f"Could not synthesize Class3; columns: {out.columns.tolist()}")
        return out, "Class3"

    raise KeyError(f"Could not locate a Class3-like column. Available columns: {df.columns.tolist()}")


def resolve_group_vendor(df: pd.DataFrame, target_name: str = "GroupVendor") -> Tuple[pd.DataFrame, bool]:
    vendor_candidates = ["GroupVendor", "crm_main_group_vendor", "group_vendor", "Group Vendor", "crm_group_vendor"]
    out, found = _resolve_column(df, target_name, vendor_candidates)
    return out, found


def resolve_product_identifiers(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str, str]:
    """Resolve ProductNumber, ProductDescription, PurchaseQuantity.

    Returns (df, product_number_col, product_description_col, purchase_quantity_col).
    Synthesis rules:
    - ProductNumber: index-based synthetic if absent.
    - ProductDescription: fallback to ProductNumber.
    - PurchaseQuantity: synthesized zeros if absent.
    """
    out = df.copy()

    prod_candidates = [
        "ProductNumber", "productnumber", "product_number", "ProductNo", "productno", "ProductID", "productid", "ProductId",
        "MaterialNumber", "materialnumber", "material_number", "Material", "ItemNumber", "itemnumber", "item_number"
    ]
    out, _ = _resolve_column(out, "ProductNumber", prod_candidates,
                             synth_func=lambda d: "SYN_" + d.index.astype(str),
                             warn_msg="Warning: Synthesized 'ProductNumber'.")

    desc_candidates = [
        "ProductDescription", "productdescription", "product_description", "Description", "Product Desc", "ProductDesc",
        "MaterialDescription", "materialdescription", "MaterialDesc", "ItemDescription", "itemdescription", "ItemDesc"
    ]
    out, _ = _resolve_column(out, "ProductDescription", desc_candidates,
                             synth_func=lambda d: d["ProductNumber"].astype(str),
                             warn_msg="Warning: Using ProductNumber as ProductDescription.")

    qty_candidates = [
        "PurchaseQuantity", "purchasequantity", "purchase_quantity", "Quantity", "Qty", "Purchase Qty", "purchase_qty", "qty"
    ]
    out, _ = _resolve_column(out, "PurchaseQuantity", qty_candidates,
                             synth_func=lambda d: 0,
                             warn_msg="Warning: Created placeholder 'PurchaseQuantity'=0.")

    return out, "ProductNumber", "ProductDescription", "PurchaseQuantity"


def resolve_all(df: pd.DataFrame) -> pd.DataFrame:
    """Composite convenience: resolve spend, class3, vendor, and product identifiers.

    Returns modified DataFrame with standardized columns.
    """
    out, spend_col = resolve_spend(df)
    out, class3_col = resolve_class3(out)
    out, _ = resolve_group_vendor(out)
    out, pn, pd, pq = resolve_product_identifiers(out)
    return out
