"""Top-level source package exports

Curated re-exports for notebook ergonomics. After migrating to the denormalized
EU super table you can simply:

    from source import fetch_super_table, compute_totals, YEAR_COLUMN_GROUPS

Includes filtered BigQuery loaders, attribute pivot helpers, total feature
builders, and dataframe-side filter utilities.
"""

# Core super table / analysis helpers
from .data_processing.analysis_utils import (
	fetch_super_table,
	fetch_super_table_for_clustering,
	pivot_attributes_wide,
	compute_totals,
	summarize_super_table,
	build_year_metrics_if_missing,
	fetch_distinct_values,
	apply_df_filters,
	# Constants / column groups
	PRODUCT_COLS,
	ATTRIBUTE_COLS,
	DEFAULT_COLUMNS,
	PURCHASE_AMOUNT_YEAR_COLS,
	SALES_QUANTITY_YEAR_COLS,
	PURCHASE_QUANTITY_YEAR_COLS,
	ORDERS_YEAR_COLS,
	YEAR_COLUMN_GROUPS,
)

# DataFrame product utilities (filters, coercion, brand map, summaries)
from .data_processing.product_utils import (
	coerce_dtypes,
	validate_has,
	filter_by_class2,
	filter_by_class3,
	filter_by_class4,
	filter_by_brand_name,
	filter_by_country_of_origin,
	filter_by_group_supplier,
	filter_not_purchase_stop,
	assert_no_purchase_stop,
	summary_stats,
	wide_to_long_year_metrics,
)

__all__ = [
	# Analysis utils
	"fetch_super_table",
	"fetch_super_table_for_clustering",
	"pivot_attributes_wide",
	"compute_totals",
	"summarize_super_table",
	"build_year_metrics_if_missing",
	"fetch_distinct_values",
	"apply_df_filters",
	# Column constants
	"PRODUCT_COLS",
	"ATTRIBUTE_COLS",
	"DEFAULT_COLUMNS",
	"PURCHASE_AMOUNT_YEAR_COLS",
	"SALES_QUANTITY_YEAR_COLS",
	"PURCHASE_QUANTITY_YEAR_COLS",
	"ORDERS_YEAR_COLS",
	"YEAR_COLUMN_GROUPS",
	# Product DF utilities
	"coerce_dtypes",
	"validate_has",
	"filter_by_class2",
	"filter_by_class3",
	"filter_by_class4",
	"filter_by_brand_name",
	"filter_by_country_of_origin",
	"filter_by_group_supplier",
	"filter_not_purchase_stop",
	"assert_no_purchase_stop",
	"summary_stats",
	"wide_to_long_year_metrics",
]
