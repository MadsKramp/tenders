"""Data Processing Module

Unified entry-point for analytical workflows over the denormalized
`super_table` including:

Core capabilities
-----------------
* Clustering analysis (two-feature economic segmentation)
* Data preprocessing & dataframe-side filtering helpers
* Feature engineering (wide attribute pivots, aggregate totals)
* Lightweight SQL script discovery & loading (`sqlScripts/`)

Public Exports
--------------
* `ClusteringPipeline` – end-to-end clustering workflow
* `fetch_super_table` – filtered BigQuery loader for the denormalized table
* `pivot_attributes_wide` – attribute row→column pivot
* `fetch_distinct_values`, `apply_df_filters` – UI/dropdown & client-side filters
* Column constants: `PRODUCT_COLS`, `ATTRIBUTE_COLS`, `DEFAULT_COLUMNS`
* `SUPER_TABLE_FQN` – fully-qualified reference to the active super table
* SQL script registry: `SQL_SCRIPT_PATHS`, `load_sql_script`

Import convenience
------------------
from source.data_processing import ClusteringPipeline, fetch_super_table, pivot_attributes_wide

Notes
-----
If the physical table name differs (e.g. `super_tender_material`) set the env
`TABLE_ID` before importing to keep `SUPER_TABLE_FQN` aligned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from .clustering_pipeline import ClusteringPipeline
from .analysis_utils import (
	fetch_super_table,
	fetch_super_table_for_clustering,
	pivot_attributes_wide,
	compute_totals,
	summarize_super_table,
	build_year_metrics_if_missing,
	fetch_distinct_values,
	apply_df_filters,
	SUPER_TABLE_FQN,
	PRODUCT_COLS,
	ATTRIBUTE_COLS,
	DEFAULT_COLUMNS,
)

# ----------------------------------------------------------------------------
# SQL script registry (auto-detect files under project sqlScripts/ directory)
# ----------------------------------------------------------------------------
_SQL_DIR = Path(__file__).resolve().parents[2] / "sqlScripts"

def _discover_sql_scripts() -> Dict[str, Path]:  # pragma: no cover - simple IO
	mapping: Dict[str, Path] = {}
	if _SQL_DIR.is_dir():
		for p in _SQL_DIR.glob("*.sql"):
			# key: file stem (without extension) e.g. 'create_purchase_data'
			mapping[p.stem] = p
	return mapping

SQL_SCRIPT_PATHS: Dict[str, Path] = _discover_sql_scripts()

def load_sql_script(name: str) -> str:
	"""Return raw SQL text for a registered script.

	Parameters
	----------
	name : str
		Stem of the file, e.g. 'create_purchase_data'. Case-sensitive.

	Raises
	------
	KeyError if the script is not found.
	"""
	path = SQL_SCRIPT_PATHS.get(name)
	if path is None:
		raise KeyError(f"SQL script '{name}' not found. Available: {sorted(SQL_SCRIPT_PATHS)}")
	return path.read_text(encoding="utf-8")

__all__ = [
	# Pipeline
	"ClusteringPipeline",
	# Super table helpers
	"fetch_super_table",
	"fetch_super_table_for_clustering",
	"pivot_attributes_wide",
	"compute_totals",
	"summarize_super_table",
	"build_year_metrics_if_missing",
	"fetch_distinct_values",
	"apply_df_filters",
	# Constants
	"SUPER_TABLE_FQN",
	"PRODUCT_COLS",
	"ATTRIBUTE_COLS",
	"DEFAULT_COLUMNS",
	# SQL registry
	"SQL_SCRIPT_PATHS",
	"load_sql_script",
]