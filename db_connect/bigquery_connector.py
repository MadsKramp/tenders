"""
BigQuery Database Connector

A reusable module for connecting to and querying Google BigQuery.
Supports both user account and service account authentication.

Usage:
    from source.db_connect.bigquery_connector import BigQueryConnector
    
    # Initialize with environment variables
    bq = BigQueryConnector()
    
    # Run a query
    df = bq.query("SELECT * FROM `project.dataset.table` LIMIT 10")
    
    # Get table info
    info = bq.get_table_info('dataset_id', 'table_id')
"""

import os
import json
import pandas as pd
from typing import Optional, Dict, List, Any
from datetime import datetime
import traceback
import sys

from google.cloud import bigquery
from google.oauth2 import service_account
from google.auth import default
from google.oauth2.credentials import Credentials
from google.auth.transport import requests
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


def check_dependencies():
    """Check if all required dependencies are installed."""
    missing_deps = []
    performance_info = []
    
    try:
        import db_dtypes
        performance_info.append("✓ db-dtypes: Enhanced BigQuery data type handling")
    except ImportError:
        missing_deps.append("db-dtypes")
        performance_info.append("✗ db-dtypes: Missing - BigQuery data types may not convert properly")
    
    try:
        import pyarrow
        performance_info.append("✓ PyArrow: Fast data conversion (10-100x faster for large datasets)")
    except ImportError:
        missing_deps.append("pyarrow")
        performance_info.append("✗ PyArrow: Missing - Conversion will be significantly slower for large datasets")
    
    # Always show performance status
    print("📊 Performance Dependencies Status:")
    for info in performance_info:
        print(f"   {info}")
    
    if missing_deps:
        deps_str = ", ".join(missing_deps)
        print(f"\n⚠️ Install missing dependencies for better performance:")
        print(f"   pip install {deps_str}")
        if "pyarrow" in missing_deps:
            print(f"   ⭐ PyArrow is especially important for large datasets!")
        print()
    else:
        print(f"✓ All performance dependencies are installed!\n")
    
    return len(missing_deps) == 0

# Check dependencies on module import
check_dependencies()


class BigQueryConnector:
    """
    A connector class for Google BigQuery operations.
    
    Handles authentication automatically and provides convenient methods
    for common BigQuery operations.
    """
    
    def __init__(self, project_id: Optional[str] = None, credentials_path: Optional[str] = None):
        """
        Initialize BigQuery connector.
        
        Args:
            project_id: Google Cloud project ID. If None, reads from PROJECT_ID env var.
            credentials_path: Path to credentials JSON file. If None, reads from 
                            GOOGLE_APPLICATION_CREDENTIALS env var.
        """
        try:
            print(f"🔧 Initializing BigQuery connector...")
            
            self.project_id = project_id or os.getenv('PROJECT_ID')
            print(self.project_id )
            self.credentials_path = credentials_path or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            
            print(f"   Project ID: {self.project_id}")
            print(f"   Credentials path: {self.credentials_path}")
            
            if not self.project_id:
                raise ValueError(
                    "PROJECT_ID must be provided either as parameter or in environment variable"
                )
            
            print(f"   Initializing client...")
            self.client = self._initialize_client()
            print(f"✓ Connected to BigQuqery project: {self.project_id}")
            
        except Exception as e:
            self._log_error("BigQueryConnector.__init__", e, f"project_id={project_id}, credentials_path={credentials_path}")
            raise
    
    def _initialize_client(self) -> bigquery.Client:
        """Initialize and return a BigQuery client with proper authentication."""
        try:
            print(f"   Checking credentials path...")
            
            if self.credentials_path and os.path.exists(self.credentials_path):
                print(f"   Credentials file found, creating client with credentials...")
                return self._create_client_with_credentials()
            else:
                if self.credentials_path:
                    print(f"   Credentials file not found at: {self.credentials_path}")
                else:
                    print(f"   No credentials path specified")
                print(f"   Using default credentials...")
                return bigquery.Client(project=self.project_id)
                
        except Exception as e:
            self._log_error("_initialize_client", e, f"credentials_path={self.credentials_path}")
            raise
    
    def _create_client_with_credentials(self) -> bigquery.Client:
        """Create BigQuery client using credentials file."""
        try:
            print(f"   Loading credentials file...")
            with open(self.credentials_path, 'r') as f:
                cred_data = json.load(f)
            
            print(f"   Credentials file loaded, analyzing format...")
            
            if cred_data.get('type') == 'service_account':
                print(f"   Detected service account credentials")
                # Service account credentials
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path,
                    scopes=['https://www.googleapis.com/auth/bigquery']
                )
                print(f"✓ Using service account credentials")
                
            elif 'client_id' in cred_data and 'refresh_token' in cred_data:
                print(f"   Detected user account credentials")
                # User account credentials
                credentials = Credentials.from_authorized_user_file(self.credentials_path)
                
                # Refresh if expired
                if credentials.expired:
                    print("   Refreshing expired credentials...")
                    credentials.refresh(requests.Request())
                
                print(f"✓ Using user account credentials")
                
            else:
                print(f"   Unknown credentials format, keys found: {list(cred_data.keys())}")
                print(f"   Falling back to default...")
                credentials, _ = default()
            
            print(f"   Creating BigQuery client...")
            client = bigquery.Client(credentials=credentials, project=self.project_id)
            print(f"   BigQuery client created successfully")
            return client
            
        except Exception as e:
            self._log_error("_create_client_with_credentials", e, f"credentials_path={self.credentials_path}")
            print("   Falling back to default credentials...")
            return bigquery.Client(project=self.project_id)
    
    def query(self, sql: str) -> Optional[pd.DataFrame]:
        """
        Execute a SQL query and return results as a pandas DataFrame.
        
        Args:
            sql: The SQL query to execute
            
        Returns:
            pandas.DataFrame with query results, or None if error occurred
        """
        try:
            print(f"🔍 Executing query...")
            print(f"   Query preview: {sql[:100]}{'...' if len(sql) > 100 else ''}")
            
            print(f"   Submitting query to BigQuery...")
            query_job = self.client.query(sql)
            
            print(f"   Waiting for query results...")
            results = query_job.result()
            
            print(f"   Converting to DataFrame...")

            df = results.to_dataframe(create_bqstorage_client=False)
            
            print(f"✓ Query executed successfully. Retrieved {len(df):,} rows.")
            return df
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Provide specific help for common dependency issues
            if "db-dtypes" in error_msg:
                print(f"✗ Missing required package for BigQuery data types:")
                print(f"   Run: pip install db-dtypes")
                print(f"   Or: pip install -r requirements.txt")
            elif "pyarrow" in error_msg:
                print(f"✗ Missing recommended package for better performance:")
                print(f"   Run: pip install pyarrow")
            else:
                self._log_error("query", e, f"SQL length: {len(sql)} chars")
            
            return None
    
    def query_to_csv(self, sql: str, filename: str, add_timestamp: bool = True) -> Optional[str]:
        """
        Execute a query and save results directly to CSV.
        
        Args:
            sql: The SQL query to execute
            filename: Output filename (without extension)
            add_timestamp: Whether to add timestamp to filename
            
        Returns:
            The actual filename used, or None if error occurred
        """
        df = self.query(sql)
        if df is not None:
            return self.save_dataframe(df, filename, add_timestamp)
        return None
    
    def save_to_table(self, df: pd.DataFrame, dataset_id: str, table_id: str, 
                      write_disposition: str = 'WRITE_APPEND', 
                      create_if_not_exists: bool = True) -> bool:
        """
        Save DataFrame to BigQuery table.
        
        Args:
            df: pandas DataFrame to save
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            write_disposition: Write mode - 'WRITE_TRUNCATE' (replace), 
                             'WRITE_APPEND' (add), or 'WRITE_EMPTY' (error if exists)
            create_if_not_exists: Whether to create table if it doesn't exist
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"💾 Saving DataFrame to BigQuery table...")
            print(f"   Target: {self.project_id}.{dataset_id}.{table_id}")
            print(f"   DataFrame shape: {df.shape}")
            print(f"   Write mode: {write_disposition}")
            
            # Check if table exists
            table_exists = self.table_exists(dataset_id, table_id)
            print(f"   Table exists: {table_exists}")
            
            if not table_exists and not create_if_not_exists:
                print(f"✗ Table does not exist and create_if_not_exists=False")
                return False
            
            # Create table reference
            table_ref = self.client.dataset(dataset_id).table(table_id)
            
            # Configure job
            job_config = bigquery.LoadJobConfig(
                write_disposition=write_disposition,
                autodetect=True if not table_exists else False
            )
            
            print(f"   Starting upload job...")
            job = self.client.load_table_from_dataframe(
                df, table_ref, job_config=job_config
            )
            
            print(f"   Waiting for job completion...")
            job.result()  # Wait for job to complete
            
            print(f"✓ Successfully saved {len(df):,} rows to {dataset_id}.{table_id}")
            
            # Verify the upload
            updated_info = self.get_table_info(dataset_id, table_id)
            if updated_info:
                print(f"   Table now contains {updated_info['num_rows']:,} total rows")
            
            return True
            
        except Exception as e:
            self._log_error("save_to_table", e, 
                          f"dataset_id={dataset_id}, table_id={table_id}, shape={df.shape}, mode={write_disposition}")
            return False
        
        
    def get_table_info(self, dataset_id: str, table_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific table.
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            
        Returns:
            Dictionary with table information, or None if error occurred
        """
        try:
            print(f"📋 Getting table info for {dataset_id}.{table_id}...")
            
            print(f"   Creating table reference...")
            table_ref = self.client.dataset(dataset_id).table(table_id)
            
            print(f"   Fetching table metadata...")
            table = self.client.get_table(table_ref)
            
            print(f"   Building table info dictionary...")
            info = {
                'project_id': self.project_id,
                'dataset_id': dataset_id,
                'table_id': table_id,
                'num_rows': table.num_rows,
                'num_columns': len(table.schema),
                'size_bytes': table.num_bytes,
                'created': table.created,
                'modified': table.modified,
                'location': table.location,
                'schema': [
                    {
                        'name': field.name, 
                        'type': field.field_type,
                        'mode': field.mode,
                        'description': field.description
                    } 
                    for field in table.schema
                ]
            }
            
            print(f"✓ Retrieved info for {dataset_id}.{table_id}")
            return info
            
        except Exception as e:
            self._log_error("get_table_info", e, f"dataset_id={dataset_id}, table_id={table_id}")
            return None

    
    def table_exists(self, dataset_id: str, table_id: str) -> bool:
        """
        Check if a table exists.
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            
        Returns:
            True if table exists, False otherwise
        """
        try:
            print(f"🔍 Checking if table {dataset_id}.{table_id} exists...")
            table_ref = self.client.dataset(dataset_id).table(table_id)
            self.client.get_table(table_ref)
            print(f"✓ Table {dataset_id}.{table_id} exists")
            return True
        except Exception as e:
            print(f"ℹ️ Table {dataset_id}.{table_id} does not exist or is not accessible")
            return False
    
    def list_datasets(self) -> List[str]:
        """
        List all datasets in the project.
        
        Returns:
            List of dataset IDs
        """
        try:
            print(f"📂 Listing datasets in project {self.project_id}...")
            datasets = list(self.client.list_datasets())
            dataset_ids = [dataset.dataset_id for dataset in datasets]
            print(f"✓ Found {len(dataset_ids)} datasets in project {self.project_id}")
            return dataset_ids
            
        except Exception as e:
            self._log_error("list_datasets", e, f"project_id={self.project_id}")
            return []
    
    def list_tables(self, dataset_id: str) -> List[str]:
        """
        List all tables in a dataset.
        
        Args:
            dataset_id: BigQuery dataset ID
            
        Returns:
            List of table IDs
        """
        try:
            print(f"📂 Listing tables in dataset {dataset_id}...")
            dataset_ref = self.client.dataset(dataset_id)
            tables = list(self.client.list_tables(dataset_ref))
            table_ids = [table.table_id for table in tables]
            print(f"✓ Found {len(table_ids)} tables in dataset {dataset_id}")
            return table_ids
            
        except Exception as e:
            self._log_error("list_tables", e, f"dataset_id={dataset_id}")
            return []
    
    def get_schema(self, dataset_id: str, table_id: str) -> Optional[List[Dict[str, str]]]:
        """
        Get just the schema of a table.
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            
        Returns:
            List of field dictionaries with name, type, mode, description
        """
        try:
            print(f"📝 Getting schema for {dataset_id}.{table_id}...")
            info = self.get_table_info(dataset_id, table_id)
            return info['schema'] if info else None
        except Exception as e:
            self._log_error("get_schema", e, f"dataset_id={dataset_id}, table_id={table_id}")
            return None
    
    def sample_table(self, dataset_id: str, table_id: str, limit: int = 5) -> Optional[pd.DataFrame]:
        """
        Get a sample of rows from a table.
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            limit: Number of rows to sample
            
        Returns:
            pandas.DataFrame with sample data
        """
        try:
            print(f"🎯 Sampling {limit} rows from {dataset_id}.{table_id}...")
            sql = f"""
                SELECT *
                FROM `{self.project_id}.{dataset_id}.{table_id}`
                LIMIT {limit}
            """
            return self.query(sql)
        except Exception as e:
            self._log_error("sample_table", e, f"dataset_id={dataset_id}, table_id={table_id}, limit={limit}")
            return None

    @staticmethod
    def save_dataframe(df: pd.DataFrame, filename: str, add_timestamp: bool = True) -> str:
        """
        Save a DataFrame to CSV with optional timestamp.
        
        Args:
            df: pandas DataFrame to save
            filename: Base filename (without extension)
            add_timestamp: Whether to add timestamp to filename
            
        Returns:
            The actual filename used
        """
        try:
            print(f"💾 Saving DataFrame to CSV...")
            print(f"   DataFrame shape: {df.shape}")
            
            if add_timestamp:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"{filename}_{timestamp}.csv"
            else:
                output_file = f"{filename}.csv"
            
            print(f"   Output file: {output_file}")
            df.to_csv(output_file, index=False)
            print(f"✓ Data saved to {output_file}")
            return output_file
            
        except Exception as e:
            print(f"✗ Error saving DataFrame: {e}")
            raise
    
    def _log_error(self, operation: str, error: Exception, additional_context: str = "") -> None:
        """
        Log detailed error information including line numbers.
        
        Args:
            operation: The operation that failed
            error: The exception that occurred
            additional_context: Additional context information
        """
        exc_type, exc_value, exc_traceback = sys.exc_info()
        
        print(f"✗ Error in {operation}:")
        print(f"   Error Type: {type(error).__name__}")
        print(f"   Error Message: {str(error)}")
        
        if additional_context:
            print(f"   Context: {additional_context}")
        
        if exc_traceback:
            # Get the last frame in our code (not in external libraries)
            tb = exc_traceback
            while tb.tb_next is not None:
                tb = tb.tb_next
            
            filename = tb.tb_frame.f_code.co_filename
            line_number = tb.tb_lineno
            function_name = tb.tb_frame.f_code.co_name
            
            # Only show if it's in our file
            if 'bigquery_connector.py' in filename:
                print(f"   Location: {function_name}() at line {line_number}")
        
        # Show full traceback in debug mode
        if os.getenv('DEBUG_MODE', '').lower() in ['true', '1', 'yes']:
            print(f"   Full traceback:")
            traceback.print_exc()
        
        print()  # Empty line for readability

    def __repr__(self) -> str:
        """String representation of the connector."""
        return f"BigQueryConnector(project_id='{self.project_id}')"



# Convenience functions for quick access
def quick_query(sql: str, project_id: Optional[str] = None, use_fast_conversion: bool = True) -> Optional[pd.DataFrame]:
    """
    Quick function to execute a query without creating a connector instance.
    
    Args:
        sql: SQL query to execute
        project_id: Optional project ID (uses env var if not provided)
        use_fast_conversion: Whether to use PyArrow for faster conversion
        
    Returns:
        pandas.DataFrame with results
    """
    try:
        print(f"🚀 Quick query execution...")
        bq = BigQueryConnector(project_id=project_id)
        return bq.query(sql, use_fast_conversion=use_fast_conversion)
    except Exception as e:
        print(f"✗ Error in quick_query: {e}")
        return None


def get_kramp_fasteners_data(limit: int = 1000) -> Optional[pd.DataFrame]:
    """
    Convenience function to get Kramp fasteners data.
    
    Args:
        limit: Number of rows to retrieve
        
    Returns:
        pandas.DataFrame with fasteners data
    """
    try:
        print(f"🔩 Getting Kramp fasteners data (limit: {limit})...")
        sql = f"""
            SELECT *
            FROM `kramp-sharedmasterdata-prd.fasteners.fasteners`
            LIMIT {limit}
        """
        return quick_query(sql)
    except Exception as e:
        print(f"✗ Error in get_kramp_fasteners_data: {e}")
        return None


