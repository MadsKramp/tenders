# Tender Material / Super Table Data Processing

Unified Python utilities and SQL scripts for building and analyzing the EU denormalized `super_table` housed in BigQuery dataset `kramp-sharedmasterdata-prd.MadsH`.

## Highlights
- Filtered BigQuery loaders (`fetch_super_table`, `fetch_super_table_for_clustering`)
- Attribute row → wide pivot (`pivot_attributes_wide`)
- Automatic total feature builders & missing year metric synthesis (`compute_totals`, `build_year_metrics_if_missing`)
- Clustering pipeline scaffold (`ClusteringPipeline`) on two economic features
- Dynamic SQL script discovery (`sqlScripts/*.sql`) with hot reload (`reload_sql_registry`)
- Column group constants for yearly metrics (`YEAR_COLUMN_GROUPS`, etc.)

## Installation / Environment
Set environment variables (or `.env`) before importing. For purchase spend analysis the authoritative fully-qualified table id is:

```
PURCHASE_TABLE_FQN=kramp-sharedmasterdata-prd.MadsH.purchase_data
PROJECT_ID=kramp-sharedmasterdata-prd   # optional if you only use PURCHASE_TABLE_FQN
BQ_LOCATION=EU
```

If you instead use separate dataset/table entries, ensure they are not combined incorrectly (avoid setting `PROJECT_ID` to the full table). The notebook now hardcodes `kramp-sharedmasterdata-prd.MadsH.purchase_data` to prevent malformed concatenations like `kramp-sharedmasterdata-prd.MadsH.purchase_data.your_dataset.your_table`.

## Quick Start (Notebook)
```python
from source import (
    fetch_super_table,
    fetch_super_table_for_clustering,
    pivot_attributes_wide,
    compute_totals,
    summarize_super_table,
    build_year_metrics_if_missing,
    YEAR_COLUMN_GROUPS,
)

# 1. Load filtered slice
df = fetch_super_table(class3=["Threaded Rods"], brand_name=["Kramp"], limit=500)

# 2. Ensure year metrics present (if transactional rows only)
df_aug = build_year_metrics_if_missing(df)

# 3. Compute totals & summarize
df_tot = compute_totals(df_aug)
summary = summarize_super_table(df_tot)
print(summary)

# 4. Pivot attributes if present
if "attribute_id" in df_tot.columns:
    wide = pivot_attributes_wide(df_tot, index_cols=("item_number",), value_col="locale_independent_values")
    print(wide.shape)
```

## Clustering Pipeline Skeleton
```python
from source.data_processing import ClusteringPipeline
pipe = ClusteringPipeline()
pipe.load_data()                # fetches filtered data based on clustering_params
pipe.prepare_features()         # builds two feature columns
pipe.optimize_clusters()        # finds optimal cluster counts
pipe.run_kmeans()               # executes KMeans clustering
```

## SQL Script Registry
```python
from source.db_connect.sql_queries import get_sql, reload_sql_registry, SQL_SCRIPTS

# Access annotated super table creation SQL
super_sql = get_sql("super_table")

# List available scripts
print(sorted(SQL_SCRIPTS))

# After adding a new file under sqlScripts/ re-scan without restarting:
reload_sql_registry()
```

## Year Column Groups
```python
from source import PURCHASE_AMOUNT_YEAR_COLS, SALES_QUANTITY_YEAR_COLS
print(PURCHASE_AMOUNT_YEAR_COLS)
print(SALES_QUANTITY_YEAR_COLS)
```

## Missing Year Metrics Helper
`build_year_metrics_if_missing` derives columns like `turnover_eur_2023`, `quantity_sold_2024`, `margin_gross_eur_2024`, `orders_2023` when your frame only has transactional rows (one per order line) with raw columns: `TurnoverEuro`, `QuantitySold`, `MarginEuro`, `ListPriceTurnoverEuro`, `OrderNumber`, `YearNumber`.

## Contributing / Next Steps
- Implement `_analyze_result` in `ClusteringPipeline` for richer metrics if required.
- Harmonize any physical table name differences (e.g. `super_tender_material`).
- Add unit tests for year metric synthesis edge cases (missing years, partial metrics).

---
Generated README snippet based on `data_processing` module docstring and current exports.
