# BigQuery Database Connector

A reusable Python module for connecting to and querying Google BigQuery with automatic authentication handling.

## Features

- **Automatic Authentication**: Supports both user account and service account credentials
- **Environment Variable Support**: Reads configuration from `.env` files
- **Pandas Integration**: Returns query results as pandas DataFrames
- **Error Handling**: Graceful error handling with informative messages
- **Convenience Methods**: Easy-to-use methods for common operations
- **CSV Export**: Built-in methods to save results to CSV files

## Quick Start

```python
from source.db_connect import BigQueryConnector, quick_query

# Initialize connector (uses environment variables)
bq = BigQueryConnector()

# Run a query
df = bq.query("SELECT * FROM `project.dataset.table` LIMIT 10")

# Get table information
info = bq.get_table_info('dataset_id', 'table_id')

# Quick one-off query
df = quick_query("SELECT COUNT(*) FROM `project.dataset.table`")
```

## Main Classes and Functions

### BigQueryConnector

The main class for BigQuery operations.

**Methods:**
- `query(sql)` - Execute SQL and return DataFrame
- `query_to_csv(sql, filename)` - Execute SQL and save to CSV
- `get_table_info(dataset_id, table_id)` - Get table metadata
- `list_datasets()` - List all datasets in project
- `list_tables(dataset_id)` - List tables in dataset
- `table_exists(dataset_id, table_id)` - Check if table exists
- `sample_table(dataset_id, table_id, limit)` - Get sample rows
- `get_schema(dataset_id, table_id)` - Get table schema
- `save_dataframe(df, filename)` - Save DataFrame to CSV

### Convenience Functions

- `quick_query(sql)` - Execute query without creating connector instance
- `get_kramp_purchase_data(limit)` - Get Kramp purchase data

## Configuration

The connector reads configuration from environment variables:

```bash
# Required
PROJECT_ID=your-project-id

# Optional - path to credentials JSON file
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

## Authentication Methods

1. **User Account Credentials** (recommended for development)
   - Run `gcloud auth application-default login`
   - Credentials stored in default location

2. **Service Account Credentials** (recommended for production)
   - Download service account JSON key
   - Set `GOOGLE_APPLICATION_CREDENTIALS` to file path

3. **Default Credentials**
   - Uses whatever credentials are available in the environment

## Examples

### Basic Usage

```python
from source.db_connect import BigQueryConnector

# Initialize
bq = BigQueryConnector()

# Simple query
df = bq.query(\"\"\"
    SELECT 
        column1, 
        COUNT(*) as count
    FROM `project.dataset.table`
    GROUP BY column1
    ORDER BY count DESC
    LIMIT 10
\"\"\")

print(df)
```

### Data Exploration

```python
# List available datasets
datasets = bq.list_datasets()
print(f"Available datasets: {datasets}")

# Get table info
info = bq.get_table_info('my_dataset', 'my_table')
print(f"Table has {info['num_rows']:,} rows and {info['num_columns']} columns")

# Sample data
sample = bq.sample_table('my_dataset', 'my_table', limit=5)
print(sample.head())
```

### Export to CSV

```python
# Query and save to CSV
output_file = bq.query_to_csv(\"\"\"
    SELECT * 
    FROM `project.dataset.table`
    WHERE date >= '2023-01-01'
\"\"\", "export_data")

print(f"Data saved to: {output_file}")
```

### Error Handling

```python
# The connector handles errors gracefully
df = bq.query("SELECT * FROM non_existent_table")
if df is not None:
    print("Query successful")
else:
    print("Query failed - check logs for details")
```

## File Structure

```
source/
└── db_connect/
    ├── __init__.py              # Package initialization
    ├── bigquery_connector.py    # Main connector class
    └── README.md               # This file
```

## Dependencies

- google-cloud-bigquery
- pandas
- python-dotenv
- google-auth
- google-oauth2

## Tips

1. **Performance**: Use `LIMIT` in your queries during development
2. **Costs**: Be mindful of BigQuery costs with large queries
3. **Security**: Never commit credentials files to version control
4. **Schema**: Use `get_schema()` to understand table structure before querying
5. **Sampling**: Use `sample_table()` to explore data before running large queries
