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
    enrichment_table: str = "kramp-sharedmasterdata-prd.MadsH.product_enrichment_table",
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
    if verbose:
        print(f"Detail fetch: {len(candidate_ids)} candidate ids (first 8): {candidate_ids[:8]}")

    # Attempt direct enrichment table pull first (materialized earlier)
    try:
        enrich_df = bq_client.query(f"SELECT * FROM `{enrichment_table}` WHERE ProductNumber IN UNNEST({_bq_array_literal(_normalize_numbers(product_numbers))})")
        if enrich_df is not None and not enrich_df.empty and "ProductNumber" in enrich_df.columns:
            # Reshape like long-form expected: pivot later in get_product_details_mapping
            cols = [c for c in enrich_df.columns if c != "ProductNumber"]
            long_records = []
            for _, row in enrich_df.iterrows():
                pn = str(row["ProductNumber"]).strip()
                for c in cols:
                    val = row[c]
                    if pd.isna(val) or str(val).strip() == "":
                        continue
                    long_records.append({"ProductNumber": pn, "detail_name": c, "detail_value": str(val).strip()})
            if long_records:
                df_enrich_long = pd.DataFrame(long_records).drop_duplicates(subset=["ProductNumber","detail_name"], keep="first")
                if verbose:
                    print(f"Enrichment table hit: {df_enrich_long['detail_name'].nunique()} attributes across {df_enrich_long['ProductNumber'].nunique()} products.")
                return df_enrich_long
            else:
                if verbose:
                    print("Enrichment table contained rows but no non-empty attribute values.")
        else:
            if verbose:
                print("Enrichment table empty or missing ProductNumber column; falling back to dynamic fetch.")
    except Exception as e:
        if verbose:
            print("Enrichment table query failed; falling back to dynamic fetch:", e)
    # De-duplicate while preserving order
    candidate_ids = list(dict.fromkeys(candidate_ids))
    # Guard against explosive prefix expansion: if too large, restrict to a single canonical prefix
    if len(candidate_ids) > 5000:
        base_nums = _normalize_numbers(product_numbers)
        candidate_ids = [f"ticGoldenItem-{b}" for b in base_nums]
        if verbose:
            print(f"Reduced candidate id list to {len(candidate_ids)} canonical prefixed ids to avoid oversized queries.")
    if not candidate_ids:
        return pd.DataFrame(columns=["ProductNumber", "detail_name", "detail_value"])

    def _run_batches(ids_list: List[str], names_filter: List[str]) -> List[pd.DataFrame]:
        """Primary long-form attribute fetch path (expects AttributeName/AttributeValue columns).

        If product_data does not have these columns (simplified schema), this path will return empty,
        triggering the wide-schema fallback implemented below.
        """
        names_literal_local = _bq_array_literal(names_filter) if names_filter else None
        frames_local: List[pd.DataFrame] = []
        # Probe schema once to see if AttributeName exists
        probe_sql = "SELECT * FROM `kramp-sharedmasterdata-prd.MadsH.product_data` LIMIT 1"
        probe_df = bq_client.query(probe_sql)
        has_long_form = probe_df is not None and not probe_df.empty and "AttributeName" in probe_df.columns and "AttributeValue" in probe_df.columns
        if not has_long_form:
            return frames_local  # empty triggers fallback
        for start in range(0, len(ids_list), chunk_size):
            batch_ids = ids_list[start:start + chunk_size]
            ids_literal = _bq_array_literal(batch_ids)
            name_clause = f"AND AttributeName IN UNNEST({names_literal_local})" if names_literal_local else ""
            sql = f"""
SELECT
  REGEXP_REPLACE(product_id, r'^(ticGoldenItem-|ticArticle-|ticItem-)', '') AS ProductNumber,
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

    # First attempt with static allowed list (long form). If empty, perform wide fallback.
    primary_frames = _run_batches(candidate_ids, ALLOWED_DETAIL_NAMES)
    if primary_frames:
        df_primary = pd.concat(primary_frames, ignore_index=True)
        if verbose:
            print(f"Long-form attributes fetched: {len(df_primary)} rows, {df_primary['detail_name'].nunique()} distinct names.")
        return df_primary.drop_duplicates(subset=["ProductNumber", "detail_name"], keep="first")

    # ------------------------------------------------------------------
    # Wide-schema fallback: product_data has one column per attribute.
    # We fetch rows and reshape to long form matching expected output.
    # ------------------------------------------------------------------
    # Chunked wide-schema fallback (avoids oversized IN clause)
    records: List[dict] = []
    import re
    def humanize(col: str) -> str:
        base = col.strip()
        if base.lower() == 'productnumber':
            return 'ProductNumber'
        # Convert snake_case to Title Case with spaces
        parts = base.replace('__', '_').split('_')
        parts = [p for p in parts if p]
        title = ' '.join(p.capitalize() for p in parts)
        return title
    allowed_norm_map = {re.sub(r"[^a-z0-9]+", "", n.lower()): n for n in ALLOWED_DETAIL_NAMES}
    for start in range(0, len(candidate_ids), chunk_size):
        batch_ids = candidate_ids[start:start+chunk_size]
        ids_literal = _bq_array_literal(batch_ids)
        wide_sql = f"""
        SELECT * FROM `kramp-sharedmasterdata-prd.MadsH.product_data`
        WHERE product_id IN UNNEST({ids_literal})
        """
        wide_df = bq_client.query(wide_sql)
        if wide_df is None or wide_df.empty:
            continue
        if "ProductNumber" in wide_df.columns:
            wide_df["ProductNumber"] = wide_df["ProductNumber"].astype(str).str.strip()
        else:
            wide_df["ProductNumber"] = wide_df["product_id"].astype(str).str.replace(r"^(ticGoldenItem-|ticArticle-|ticItem-)", "", regex=True)
        wide_df = wide_df[wide_df["ProductNumber"].astype(str).str.strip() != ""]
        if wide_df.empty:
            continue
        # Attribute columns = all except identifiers
        attr_cols = [c for c in wide_df.columns if c not in {"product_id", "ProductNumber"}]
        # Build normalized map of existing columns
        existing_norm = {re.sub(r"[^a-z0-9]+", "", c.lower()): c for c in attr_cols}
        for _, row in wide_df.iterrows():
            prod = str(row["ProductNumber"]).strip()
            for norm_key, real_col in existing_norm.items():
                val = row[real_col]
                if pd.isna(val) or str(val).strip() == "":
                    continue
                display_name = allowed_norm_map.get(norm_key, humanize(real_col))
                records.append({
                    "ProductNumber": prod,
                    "detail_name": display_name,
                    "detail_value": str(val).strip()
                })
    if records:
        df_wide_long = pd.DataFrame(records)
        if verbose:
            print(f"Wide-schema fallback: {len(df_wide_long)} attribute rows collected across {df_wide_long['ProductNumber'].nunique()} products.")
        return df_wide_long.drop_duplicates(subset=["ProductNumber", "detail_name"], keep="first")

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
                    if verbose:
                        print(f"Secondary fetch (discovered attributes): {len(df_secondary)} rows.")
                    return df_secondary.drop_duplicates(subset=["ProductNumber", "detail_name"], keep="first")
        else:
            if verbose:
                print("Diagnostic: no attributes found for sampled product ids.")

    # Final empty result
    if verbose:
        print("No product attribute details found after all strategies.")
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
