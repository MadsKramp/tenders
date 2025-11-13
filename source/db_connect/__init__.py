from __future__ import annotations
import logging
from importlib.metadata import version as _v, PackageNotFoundError

# Re-exports: import from `src` instead of deep utils paths
from .utils.bq import make_client, run_sql
from .utils.io import read_or_query
from .utils.validate import expect_columns, expect_non_empty
from .utils.fe import product_agg_for_clustering

__all__ = [
    "make_client",
    "run_sql",
    "read_or_query",
    "expect_columns",
    "expect_non_empty",
    "product_agg_for_clustering",
]

# Avoid "No handler found" warnings; user configures logging in notebook if desired
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Optional: version hint (works if you package/install; harmless otherwise)
try:
    __version__ = _v("spend-analysis")
except PackageNotFoundError:
    __version__ = "0.0.0"
