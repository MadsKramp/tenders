"""Build a materialized enrichment table in BigQuery for selected ProductNumbers.

Creates or replaces `kramp-sharedmasterdata-prd.MadsH.product_enrichment_table` by
filtering `product_data` to the provided product numbers (handling known prefixes).
"""
from __future__ import annotations
from typing import Iterable, List

PREFIXES = ["ticGoldenItem-", "ticArticle-", "ticItem-"]

def _normalize_numbers(product_numbers: Iterable[str]) -> List[str]:
    base: List[str] = []
    for p in product_numbers:
        if p is None:
            continue
        s = str(p).strip()
        if s:
            base.append(s)
    # unique preserve order
    return list(dict.fromkeys(base))

def _bq_array_literal(values: List[str]) -> str:
    def q(v: str) -> str:
        return "'" + v.replace("'", "''") + "'"
    return "[" + ",".join(q(v) for v in values) + "]"

def build_product_enrichment_table(bq, product_numbers: Iterable[str],
                                   destination: str = "kramp-sharedmasterdata-prd.MadsH.product_enrichment_table",
                                   source: str = "kramp-sharedmasterdata-prd.MadsH.product_data") -> None:
    nums = _normalize_numbers(product_numbers)
    if not nums:
        print("No product numbers provided for enrichment table build.")
        return
    # Build candidate prefixed ids for matching product_id column, plus unprefixed for regexp replace
    prefixed: List[str] = []
    for n in nums:
        for pref in PREFIXES:
            prefixed.append(f"{pref}{n}")
    prefixed = list(dict.fromkeys(prefixed))
    # Array literals
    arr_prefixed = _bq_array_literal(prefixed)
    arr_unprefixed = _bq_array_literal(nums)
    sql = f"""
    CREATE OR REPLACE TABLE `{destination}` AS
    SELECT
      REGEXP_REPLACE(product_id, r'^(ticGoldenItem-|ticArticle-|ticItem-)', '') AS ProductNumber,
      * EXCEPT(product_id)  -- keep all other columns
    FROM `{source}`
    WHERE product_id IN UNNEST({arr_prefixed})
       OR REGEXP_REPLACE(product_id, r'^(ticGoldenItem-|ticArticle-|ticItem-)', '') IN UNNEST({arr_unprefixed})
    """
    try:
        bq.query(sql)
        print(f"Enrichment table rebuilt with {len(nums)} base product numbers -> {len(prefixed)} prefixed ids.")
    except Exception as e:
        print("Failed to build enrichment table:", e)

def table_exists(bq, table: str) -> bool:
    try:
        probe = bq.query(f"SELECT 1 FROM `{table}` LIMIT 1")
        return probe is not None
    except Exception:
        return False
