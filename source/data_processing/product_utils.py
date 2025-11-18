# product_utils.py
from typing import Iterable, List, Sequence
import pandas as pd

ALLOWED_DETAIL_NAMES: List[str] = [
    "Unit", "Head shape", "Thread type", "Head height",
    "Head outside diameter (width)", "Quality", "Surface treatment",
    "Material", "DIN Standard", "Weight per 100 pcs", "Content in sales unit",
    "Thread diameter", "Length", "Height", "Total height", "Width",
    "ISO Standard", "Inside diameter", "Outside diameter", "Thickness",
    "Designed for thread", "Total length", "Head type", "Thread length",
]

def _normalize_numbers(product_numbers: Iterable) -> List[str]:
    if product_numbers is None:
        return []
    ser = pd.Series(product_numbers).dropna().astype(str).str.strip()
    return [v for v in ser.unique().tolist() if v]

PREFIX_CANDIDATES: Sequence[str] = [
    "ticGoldenItem-",
    "ticArticle-",
    "ticItem-",
    ""  # unprefixed fallback
]

def _build_candidate_ids(product_numbers: Iterable) -> List[str]:
    base = _normalize_numbers(product_numbers)
    out: List[str] = []
    for b in base:
        for pref in PREFIX_CANDIDATES:
            out.append(f"{pref}{b}")
    return out

def _bq_array_literal(values: List[str]) -> str:
    """
    Build a BigQuery ARRAY<STRING> literal:
      ["a","b"] -> ["a","b"]  -> in SQL: ['a','b']
    Escapes single quotes inside values.
    """
    def q(s: str) -> str:
        # Escape single quotes as two single quotes (BigQuery standard SQL)
        return "'" + s.replace("'", "''") + "'"
    return "[" + ",".join(q(v) for v in values) + "]"

def fetch_product_details(
    bq_client,
    product_numbers: pd.Series,
    chunk_size: int = 1000,
    dynamic_attributes: bool = True,
    attribute_sample_limit: int = 50,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch product attribute details for given product numbers with multi-prefix fallback.

    Process:
    1. Generate candidate product_ids using known prefixes + unprefixed.
    2. Query in chunks filtering to ALLOWED_DETAIL_NAMES.
    3. If zero rows and dynamic_attributes=True, perform a diagnostic query (limited set of product_ids)
       WITHOUT the AttributeName filter to discover available attribute names; intersect with ALLOWED_DETAIL_NAMES.
    4. Requery using discovered intersection if not empty.
    5. Return long form rows for downstream pivot.
    """
    if bq_client is None:
        raise ImportError("BigQuery client is missing.")
    candidate_ids = _build_candidate_ids(product_numbers)
    if not candidate_ids:
        return pd.DataFrame(columns=["ProductNumber", "detail_name", "detail_value"])

    def _run_batches(ids_list: List[str], names_filter: List[str]) -> List[pd.DataFrame]:
        names_literal_local = _bq_array_literal(names_filter) if names_filter else None
        frames_local: List[pd.DataFrame] = []
        for start in range(0, len(ids_list), chunk_size):
            batch_ids = ids_list[start:start + chunk_size]
            ids_literal = _bq_array_literal(batch_ids)
            name_clause = f"AND AttributeName IN UNNEST({names_literal_local})" if names_literal_local else ""
            sql = f"""
            SELECT
              REPLACE(product_id, 'ticGoldenItem-', '') AS ProductNumber,
              AttributeName AS detail_name,
              AttributeValue AS detail_value
            FROM `kramp-sharedmasterdata-prd.MadsH.product_data`
            WHERE product_id IN UNNEST({ids_literal})
              {name_clause}
            """
            df_part = bq_client.query(sql)
            if df_part is None or df_part.empty:
                continue
            df_part["ProductNumber"] = df_part["ProductNumber"].astype(str).str.strip()
            df_part["detail_name"] = df_part["detail_name"].astype(str).str.strip()
            df_part["detail_value"] = df_part["detail_value"].astype(str).str.strip()
            df_part = df_part[df_part["ProductNumber"] != ""]
            frames_local.append(df_part)
        return frames_local

    # First attempt with static allowed list
    primary_frames = _run_batches(candidate_ids, ALLOWED_DETAIL_NAMES)
    if primary_frames:
        df_primary = pd.concat(primary_frames, ignore_index=True)
        return df_primary.drop_duplicates(subset=["ProductNumber", "detail_name"], keep="first")

    # Diagnostic attribute name discovery
    if dynamic_attributes:
        sample_ids = candidate_ids[:attribute_sample_limit]
        diag_frames = _run_batches(sample_ids, [])  # no name filter
        if diag_frames:
            df_diag = pd.concat(diag_frames, ignore_index=True)
            discovered = sorted(df_diag["detail_name"].dropna().unique().tolist())
            intersect = [n for n in discovered if n in ALLOWED_DETAIL_NAMES]
            if verbose:
                print(f"Diagnostic: discovered {len(discovered)} attribute names; intersection with allowed list: {len(intersect)}.")
            if not intersect and discovered:
                # fallback: take top N frequent discovered attributes
                top_counts = (df_diag["detail_name"].value_counts().head(10).index.tolist())
                if verbose:
                    print(f"No intersection with allowed list; falling back to top discovered attributes: {top_counts}")
                intersect = top_counts
            if intersect:
                # Re-run full fetch with intersect list
                secondary_frames = _run_batches(candidate_ids, intersect)
                if secondary_frames:
                    df_secondary = pd.concat(secondary_frames, ignore_index=True)
                    return df_secondary.drop_duplicates(subset=["ProductNumber", "detail_name"], keep="first")
        else:
            if verbose:
                print("Diagnostic: no attributes found for sampled product ids.")

    # Final empty result
    return pd.DataFrame(columns=["ProductNumber", "detail_name", "detail_value"])

def pivot_product_details(df_details: pd.DataFrame) -> pd.DataFrame:
    if df_details is None or df_details.empty:
        return pd.DataFrame(columns=["ProductNumber"] + ALLOWED_DETAIL_NAMES)

    df_pivoted = (
        df_details.pivot_table(
            index="ProductNumber",
            columns="detail_name",
            values="detail_value",
            aggfunc="first",
        ).reset_index()
    )
    df_pivoted.columns.name = None
    df_pivoted.columns = [str(c) for c in df_pivoted.columns]

    # Ensure stable schema
    for col in ALLOWED_DETAIL_NAMES:
        if col not in df_pivoted.columns:
            df_pivoted[col] = pd.NA
    return df_pivoted[["ProductNumber"] + ALLOWED_DETAIL_NAMES]

def get_product_details_mapping(bq_client, product_numbers: pd.Series) -> pd.DataFrame:
    return pivot_product_details(fetch_product_details(bq_client, product_numbers))


# ---- Placeholder field value utilities (kept minimal/clean) ----
def format_units(value):
    try:
        return f"{value:,.2f} units"
    except (ValueError, TypeError):
        return "Invalid value"

def parse_units(formatted_str):
    try:
        numeric_str = formatted_str.replace(" units", "").replace(",", "")
        return float(numeric_str)
    except (ValueError, AttributeError):
        return None

def is_valid_units_format(formatted_str):
    import re
    return bool(re.match(r'^\d{1,3}(,\d{3})*(\.\d{2})? units$', formatted_str))

def preprocess_field_values(df: pd.DataFrame) -> pd.DataFrame:
    return df

def harmonize_field_values(df: pd.DataFrame) -> pd.DataFrame:
    return df

def prepare_field_values(df: pd.DataFrame) -> pd.DataFrame:
    df = preprocess_field_values(df)
    df = harmonize_field_values(df)
    return df
