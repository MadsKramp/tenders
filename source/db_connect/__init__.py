"""
BigQuery Database Connection Package

This package provides easy-to-use connectors for Google BigQuery.

Main Components:
- BigQueryConnector: Main class for BigQuery operations
- quick_query: Convenience function for one-off queries
- get_kramp_fasteners_data: Specific function for Kramp data

Usage:
    from db_connect import BigQueryConnector, quick_query

    bq = BigQueryConnector()
    df = bq.query("SELECT * FROM table")

    df = quick_query("SELECT * FROM table")
"""


from .bigquery_connector import (
    BigQueryConnector,
    quick_query
)

__version__ = "1.0.0"
__author__ = "Kramp Data Team"

__all__ = [
    'BigQueryConnector',
    'quick_query', 
]
