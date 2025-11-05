"""product_utils.py — Kramp product utilities

Pure DataFrame utilities for product data. I/O lives in gbq_io.py.

Includes:
- Validation + light dtype coercion (pyarrow strings).
- Filters (restricted): Class2, Class3, Class4, BrandName, Country of Origin, Group Supplier.
- Brand standardization using a synonym map.
- Year-metric helpers (wide→long) and simple group summaries.
- Optional GBQ loader thin-wrapper delegating to gbq_io.select_df().

Default scheduled table (Monday refresh):
  kramp-sharedmasterdata-prd.kramp_sharedmasterdata_customquery
  .TBL__ProductManagment__CategoryAnalysisMachinery__TotalQuery
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import csv
import re
import numpy as np
import pandas as pd

# ---- BigQuery delegation (use your connector) --------------------------------
from bigquery_connector import select_df as _bq_select_df, DEFAULT_FQN, ALT_VIEW_FQN  # type: ignore

# ---- Public API --------------------------------------------------------------
__all__ = [
    # I/O
    "read_from_bigquery", "cache_to_parquet", "read_parquet_cache",
    # Validation / coercion
    "validate_has", "coerce_dtypes",
    # Filters (Class2/3/4/BrandName + CountryOfOrigin + GroupSupplier)
    "filter_by",
    "filter_by_class2", "filter_by_class3", "filter_by_class4", "filter_by_brand_name",
    "filter_by_country_of_origin", "filter_by_group_supplier",
    "filter_not_purchase_stop", "assert_no_purchase_stop",
    # Brand canonicalization
    "BrandMap", "standardize_brand",
    # Summaries / year metrics
    "summary_stats", "list_year_cols", "wide_to_long_year_metrics",
    # Exporters
    "export_csv", "export_csv_excel",
    # Constants
    "DEFAULT_FQN", "ALT_VIEW_FQN",
]

# ---- Minimal schema guard ----------------------------------------------------
BASE_REQUIRED: set[str] = {
    "pk_itemnumber",
    "item_description",
    "category",
    "category_code",
    "class_2",
    "class_3",
    "brand_name",
}

TEXT_ARROW_DTYPE = "string[pyarrow]"
YEAR_PREFIXES = (
    "turnover_eur_", "orders_", "quantity_sold_", "avg_price_eur_", "avg_order_value_eur_",
    "margin_gross_eur_", "margin_gross_perc_",
    "customers_total_", "sold_to_customers_new_", "customers_repeat_", "customer_ordered_multiple_times_",
    "average_stock_", "stock_rotation_units_",
)

# ---- Small utilities ---------------------------------------------------------
def validate_has(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

def _norm_ci(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.casefold()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _norm_series_ci(ser: pd.Series) -> pd.Series:
    return ser.fillna("").astype(str).map(_norm_ci)

# ---- Dtype coercion (safe, fast) --------------------------------------------
def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # String-like columns commonly used downstream
    text_like = [
        "pk_itemnumber","item_description","category","category_code",
        "class_2","class_3","class_4","class_4_id",
        "brand_name","ptq_product_range_type",
        "machine_make_clean","machinery_group_clean","machinery_group_rev",
        "machinery_make_clean_rev","machinery_make_group",
        "tag_webshop_active","tag_item_age","tag_has_turnover","tag_has_oe_number",
        "tag_has_richard","tag_has_stock","tag_has_makemodel","stop_purchase_ind",
        "countries_webshop_ind_active",
        "level0","level1","level2","level3","level4",
        "machinery_type_1","machinery_type_2","primary_oem",
        "supplier_name_string","supplier_most_purchase",
        "turnover_share_csbuCountry_2021","turnover_share_customer_business_type_2021",
        "turnover_share_csbuCountry_2022","turnover_share_customer_business_type_2022",
        "turnover_share_csbuCountry_2023","turnover_share_customer_business_type_2023",
        "turnover_share_csbuCountry_2024","turnover_share_customer_business_type_2024",
        "active_country_names_2021","active_country_names_2022","active_country_names_2023",
        "active_country_names_2024","active_country_names_2025",
        "warehouse_with_stock",
        "STEP_CountryOfOrigin",  # NEW
    ]
    for c in (set(text_like) & set(out.columns)):
        out[c] = out[c].astype(TEXT_ARROW_DTYPE)

    # Numerics
    num_like = ["count_countries_webshop_active", "count_warehouse_with_stock",
                "stock_on_hand_units", "stock_on_hand_eur", "search_success_rate"]
    for c in (set(num_like) & set(out.columns)):
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Year-suffixed numeric columns
    for pref in YEAR_PREFIXES:
        for col in (c for c in out.columns if c.startswith(pref)):
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Dates
    for dc in ("creation_date", "plm_launch_date", "plm_expiration_date"):
        if dc in out.columns:
            out[dc] = pd.to_datetime(out[dc], errors="coerce", utc=False).dt.date

    return out

# ---- I/O thin wrapper (delegates to gbq_io) ---------------------------------
def read_from_bigquery(
    fqn: str = DEFAULT_FQN,
    project_id: Optional[str] = None,
    where_sql: Optional[str] = None,
    limit: Optional[int] = None,
    columns: Optional[Sequence[str]] = None,
    location: str = "EU",
    maximum_bytes_billed: Optional[int] = None,
    timeout: int = 600,
) -> pd.DataFrame:
    df = _bq_select_df(
        fqn=fqn,
        columns=columns,
        where_sql=where_sql,
        limit=limit,
        project_id=project_id,
        location=location,
        maximum_bytes_billed=maximum_bytes_billed,
        timeout=timeout,
    )
    validate_has(df, BASE_REQUIRED & set(df.columns))
    return coerce_dtypes(df)

def cache_to_parquet(df: pd.DataFrame, path: str | Path, overwrite: bool = True) -> Path:
    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(p)
    df.to_parquet(p, index=False)
    return p

def read_parquet_cache(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)

# ---- Filters (ONLY Class2, Class3, Class4, BrandName, CountryOfOrigin, GroupSupplier) --
_ALLOWED_FIELDS_MAP = {
    "class_2": {"class_2", "class2", "Class2", "Class_2"},
    "class_3": {"class_3", "class3", "Class3", "Class_3"},
    "class_4": {"class_4", "class4", "Class4", "Class_4"},
    "brand_name": {"brand_name", "brandname", "BrandName"},
    # NEW fields/aliases
    "STEP_CountryOfOrigin": {
        "STEP_CountryOfOrigin", "country_of_origin", "CountryOfOrigin", "origin", "Country of Origin"
    },
    "supplier_name_string": {
        "supplier_name_string", "group_supplier", "GroupSupplier", "Group Supplier",
        "supplier", "group vendor", "group_vendor", "GroupVendor"
    },
}

def _resolve_filter_field(field: str) -> str:
    f = field.strip().casefold()
    for canon, aliases in _ALLOWED_FIELDS_MAP.items():
        if f in {a.casefold() for a in aliases}:
            return canon
    allowed = "Class2, Class3, Class4, BrandName, Country of Origin, Group Supplier"
    raise ValueError(f"Unsupported field. Allowed: {allowed} "
                     "(or GBQ: class_2, class_3, class_4, brand_name, STEP_CountryOfOrigin, supplier_name_string).")

def _ci_set(values: Iterable[str]) -> set[str]:
    return {_norm_ci(v) for v in values}

def filter_by(
    df: pd.DataFrame,
    field: str,
    values: Iterable[str],
    negate: bool = False,
) -> pd.DataFrame:
    """Case-insensitive filter limited to the allowed fields map."""
    canon = _resolve_filter_field(field)
    if canon not in df.columns:
        raise ValueError(f"Column '{canon}' not in DataFrame.")
    want = _ci_set(values)
    mask = _norm_series_ci(df[canon]).isin(want)
    return df.loc[~mask if negate else mask].copy()

def filter_by_class2(df: pd.DataFrame, values: Iterable[str], negate: bool = False) -> pd.DataFrame:
    return filter_by(df, "class_2", values, negate)

def filter_by_class3(df: pd.DataFrame, values: Iterable[str], negate: bool = False) -> pd.DataFrame:
    return filter_by(df, "class_3", values, negate)

def filter_by_class4(df: pd.DataFrame, values: Iterable[str], negate: bool = False) -> pd.DataFrame:
    return filter_by(df, "class_4", values, negate)

def filter_by_brand_name(df: pd.DataFrame, values: Iterable[str], negate: bool = False) -> pd.DataFrame:
    return filter_by(df, "brand_name", values, negate)

def filter_by_country_of_origin(df: pd.DataFrame, values: Iterable[str], negate: bool = False) -> pd.DataFrame:
    """Case-insensitive filter on STEP_CountryOfOrigin."""
    return filter_by(df, "STEP_CountryOfOrigin", values, negate)

def filter_by_group_supplier(df: pd.DataFrame, values: Iterable[str], negate: bool = False) -> pd.DataFrame:
    """Case-insensitive filter on supplier_name_string (group supplier)."""
    return filter_by(df, "supplier_name_string", values, negate)

# ---- Brand canonicalization --------------------------------------------------
@dataclass(frozen=True)
class BrandMap:
    """Canonical brand -> list of synonyms (matched against brand_name)."""
    synonyms: Mapping[str, Sequence[str]]

@lru_cache(maxsize=256)
def _canon_key(canon: str, frags: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    return canon, frags

def standardize_brand(
    df: pd.DataFrame,
    brand_map: BrandMap,
    src_col: str = "brand_name",
    out_col: str = "brand",
    overwrite: bool = False,
) -> pd.DataFrame:
    """Map brand_name to canonical labels; leaves unmatched as original."""
    validate_has(df, [src_col])
    out = df.copy()
    if out_col in out.columns and not overwrite:
        return out

    lut: dict[str, str] = {}
    for canon, frags in brand_map.synonyms.items():
        _canon_key(canon, tuple(frags))
        lut[_norm_ci(canon)] = canon
        for f in frags:
            if f:
                lut[_norm_ci(f)] = canon

    def _map_one(x: str) -> str:
        nx = _norm_ci(x)
        return lut.get(nx, x)

    out[out_col] = out[src_col].astype(str).map(_map_one).astype(TEXT_ARROW_DTYPE)
    return out

# ---- Summaries / year-metric helpers ----------------------------------------
def summary_stats(
    df: pd.DataFrame,
    by: Sequence[str] = ("class_2", "class_3", "brand_name"),
    count_col: str = "n",
) -> pd.DataFrame:
    """Counts + share_total (+ share_in_class_2 if applicable)."""
    validate_has(df, [c for c in by if c in ("class_2", "class_3", "brand_name")] or ["pk_itemnumber"])
    g = df.groupby(list(by), dropna=False).size().rename(count_col).reset_index()
    total = float(g[count_col].sum())
    g["share_total"] = g[count_col] / total if total else 0.0
    if "class_2" in by:
        g["share_in_class_2"] = g[count_col] / g.groupby("class_2")[count_col].transform("sum")
    return g.sort_values(count_col, ascending=False).reset_index(drop=True)

def list_year_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    return sorted([c for c in df.columns if c.startswith(prefix) and c[len(prefix):].isdigit()])

def wide_to_long_year_metrics(
    df: pd.DataFrame,
    prefixes: Sequence[str],
    id_vars: Sequence[str] = ("pk_itemnumber",),
    var_name: str = "metric",
    year_name: str = "year",
    value_name: str = "value",
) -> pd.DataFrame:
    """Melt one or more year-suffixed metric groups into tidy long format."""
    cols: list[str] = []
    for p in prefixes:
        cols.extend(list_year_cols(df, p))
    if not cols:
        raise ValueError("No year-suffixed columns found for provided prefixes.")

    parts: list[pd.DataFrame] = []
    for p in prefixes:
        pcols = list_year_cols(df, p)
        if not pcols:
            continue
        tmp = df[id_vars + pcols].melt(id_vars=id_vars, var_name=year_name, value_name=value_name)
        tmp[var_name] = p
        tmp["year"] = tmp[year_name].str.extract(r"(\d{4})").astype("Int64")
        tmp = tmp.drop(columns=[year_name])
        parts.append(tmp)

    return pd.concat(parts, axis=0, ignore_index=True)

# ---- CSV exporters -----------------------------------------------------------
def export_csv(
    df: pd.DataFrame,
    path: str | Path,
    *,
    columns: Optional[Sequence[str]] = None,
    sep: str = ",",
    encoding: str = "utf-8",
    gzip: bool | None = None,                 # True -> .csv.gz
    index: bool = False,
    na_rep: str = "",
    float_format: str | None = None,          # e.g. "%.2f"
    date_format: str | None = "%Y-%m-%d",
    quoting: int = csv.QUOTE_MINIMAL,
    chunksize: int | None = 1_000_000,        # stream large files
) -> Path:
    """
    Robust CSV export with sane defaults; makes parent dirs; optional gzip.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # decide compression
    if gzip is None:
        compression = "gzip" if p.suffix == ".gz" else None
    else:
        compression = "gzip" if gzip else None
        if gzip and p.suffix != ".gz":
            p = p.with_suffix(p.suffix + ".gz")

    data = df if columns is None else df.loc[:, columns]
    data.to_csv(
        p,
        sep=sep,
        encoding=encoding,
        index=index,
        na_rep=na_rep,
        float_format=float_format,
        date_format=date_format,
        compression=compression,
        quoting=quoting,
        lineterminator="\n",
        chunksize=chunksize,
    )
    return p

def export_csv_excel(
    df: pd.DataFrame,
    path: str | Path,
    *,
    columns: Optional[Sequence[str]] = None,
    sep: str = ";",                 # Excel EU-friendly
    encoding: str = "utf-8-sig",    # BOM so Excel opens UTF-8 correctly
    decimal: str = ",",             # print decimals with comma
    index: bool = False,
    na_rep: str = "",
    float_format: str | None = None,
    date_format: str | None = "%Y-%m-%d",
    chunksize: int | None = 1_000_000,
) -> Path:
    """
    Excel-friendly CSV for EU locales (semicolon; UTF-8 BOM; decimal comma).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = df if columns is None else df.loc[:, columns]
    data.to_csv(
        p,
        sep=sep,
        encoding=encoding,
        index=index,
        na_rep=na_rep,
        float_format=float_format,
        date_format=date_format,
        decimal=decimal,
        lineterminator="\n",
        chunksize=chunksize,
    )
    return p

# --- Purchase stop filters ----------------------------------------------------
def filter_not_purchase_stop(df: pd.DataFrame, *, include_unknown: bool = False) -> pd.DataFrame:
    """
    Keep only items **not on purchase stop**.

    Parameters
    ----------
    include_unknown : bool, default False
        If False (strict): keep only stop_purchase_ind == 'N'.
        If True  (lenient): keep rows where stop_purchase_ind != 'Y'
        (i.e., includes 'Unknown' and nulls).
    """
    validate_has(df, ["stop_purchase_ind"])
    s = df["stop_purchase_ind"].astype(str).str.strip().str.upper()
    mask = s.ne("Y") if include_unknown else s.eq("N")
    return df.loc[mask].copy()

def assert_no_purchase_stop(df: pd.DataFrame) -> None:
    """Raise if any item is on purchase stop (stop_purchase_ind == 'Y')."""
    validate_has(df, ["stop_purchase_ind"])
    n = df["stop_purchase_ind"].astype(str).str.strip().str.upper().eq("Y").sum()
    if n:
        raise AssertionError(f"{n:,} items are on purchase stop (stop_purchase_ind='Y').")

# ---- Self-test ---------------------------------------------------------------
if __name__ == "__main__":
    demo = pd.DataFrame(
        {
            "pk_itemnumber": ["X1","X2","X3","X4"],
            "item_description": ["A","B","C","D"],
            "category": ["Machinery"]*4,
            "category_code": ["CAT01"]*4,
            "class_2": ["Tractors","Tractors","Implements","Implements"],
            "class_3": ["Engines","Cab","Hydraulics","Hydraulics"],
            "class_4": ["1234 - Filters","5678 - Cab parts","9999 - Hose","9999 - Hose"],
            "brand_name": ["John Deere","KUBOTA","Unknown","JD"],
            "supplier_name_string": ["Kubota Corp","ACME","JD Group","ACME"],
            "STEP_CountryOfOrigin": ["NL","DE","NL","PL"],
            "turnover_eur_2024": [1000, 0, 50, 500],
            "stop_purchase_ind": ["N","N","Y","Unknown"],
        }
    )
    demo = coerce_dtypes(demo)
    # Filters
    _ = filter_by_class2(demo, ["Tractors"])
    _ = filter_by_brand_name(demo, ["John Deere","jd"])
    _ = filter_by_country_of_origin(demo, ["nl", "PL"])
    _ = filter_by_group_supplier(demo, ["ACME"])
    # Purchase stop checks
    _ = filter_not_purchase_stop(demo, include_unknown=True)
    try:
        assert_no_purchase_stop(demo)
    except AssertionError as e:
        print("assert_no_purchase_stop:", e)
