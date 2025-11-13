"""
BigQuery Database Connection Package

This package provides easy-to-use connectors for Google BigQuery.

Main Components:
- BigQueryConnector: Main class for BigQuery operations
- quick_query: Convenience function for one-off queries
- get_kramp_fasteners_data: Specific function for Kramp data

Usage:
    from source.db_connect import BigQueryConnector, quick_query
    
    # Use the main connector
    bq = BigQueryConnector()
    df = bq.query("SELECT * FROM table")
    
    # Or use quick functions
    df = quick_query("SELECT * FROM table")
"""

from .bigquery_connector import (
    BigQueryConnector,
    quick_query
)

__version__ = "1.0.0"
__author__ = "Kramp Data Team"

__all__ = [
    'bigquery_connector',
    'quick_query', 
]