from __future__ import annotations
from .bq import make_client, run_sql
from .io import read_or_query
from .validate import expect_columns, expect_non_empty
from .fe import product_agg_for_clustering
from .spend import (
    DISPLAY_NAME,
    vendor_topN_by_year_sql,
    topN_by_year,
    top20_by_year,
    overview_topN,
    pivot_topN_yearly,
    fmt_eur,
    plot_scatter_eur,
)

__all__ = [
    "make_client", "run_sql", "read_or_query",
    "expect_columns", "expect_non_empty", "product_agg_for_clustering",
    "DISPLAY_NAME",
    "vendor_topN_by_year_sql", "topN_by_year", "top20_by_year",
    "overview_topN", "pivot_topN_yearly",
    "fmt_eur", "plot_scatter_eur",
]
