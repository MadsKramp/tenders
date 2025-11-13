from __future__ import annotations
import logging

from .utils.bq import make_client, run_sql
from .utils.io import read_or_query
from .utils.validate import expect_columns, expect_non_empty
from .utils.fe import product_agg_for_clustering
from .utils.spend import (
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

logging.getLogger(__name__).addHandler(logging.NullHandler())
