"""BigQuery Database Connector (ASCII-safe logging)."""

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

load_dotenv()

def _safe_print(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        ascii_msg = ''.join(ch if ord(ch) < 128 else '?' for ch in msg)
        try:
            print(ascii_msg)
        except Exception:
            pass

def check_dependencies() -> bool:
    missing = []
    info_lines = []
    try:
        import db_dtypes  # noqa: F401
        info_lines.append("OK db-dtypes: Enhanced BigQuery data type handling")
    except ImportError:
        missing.append("db-dtypes")
        info_lines.append("MISSING db-dtypes: Limited type handling")
    try:
        import pyarrow  # noqa: F401
        info_lines.append("OK pyarrow: Fast conversion")
    except ImportError:
        missing.append("pyarrow")
        info_lines.append("MISSING pyarrow: Slower conversion")
    _safe_print("Performance dependency status:")
    for line in info_lines:
        _safe_print("  - " + line)
    if missing:
        _safe_print("Install for best performance: pip install " + " ".join(missing))
    else:
        _safe_print("All optional performance dependencies present.")
    return not missing

check_dependencies()

class BigQueryConnector:
    def __init__(self, project_id: Optional[str] = None, credentials_path: Optional[str] = None):
        try:
            _safe_print("Initializing BigQuery connector...")
            self.project_id = project_id or os.getenv('PROJECT_ID')
            self.credentials_path = credentials_path or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            _safe_print(f"   Project ID: {self.project_id}")
            _safe_print(f"   Credentials path: {self.credentials_path}")
            if not self.project_id:
                raise ValueError("PROJECT_ID must be provided via param or env var")
            _safe_print("   Initializing client...")
            self.client = self._initialize_client()
            _safe_print(f"Connected to BigQuery project: {self.project_id}")
        except Exception as e:
            self._log_error("BigQueryConnector.__init__", e, f"project_id={project_id}, credentials_path={credentials_path}")
            raise

    def _initialize_client(self) -> bigquery.Client:
        try:
            _safe_print("   Checking credentials path...")
            if self.credentials_path and os.path.exists(self.credentials_path):
                _safe_print("   Credentials file found; using explicit credentials")
                return self._create_client_with_credentials()
            else:
                if self.credentials_path:
                    _safe_print(f"   Credentials file not found at: {self.credentials_path}")
                else:
                    _safe_print("   No credentials path specified")
                _safe_print("   Using default credentials")
                return bigquery.Client(project=self.project_id)
        except Exception as e:
            self._log_error("_initialize_client", e, f"credentials_path={self.credentials_path}")
            raise

    def _create_client_with_credentials(self) -> bigquery.Client:
        try:
            _safe_print("   Loading credentials file...")
            with open(self.credentials_path, 'r') as f:
                cred_data = json.load(f)
            _safe_print("   Credentials file loaded; analyzing format")
            if cred_data.get('type') == 'service_account':
                _safe_print("   Detected service account credentials")
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path,
                    scopes=['https://www.googleapis.com/auth/bigquery']
                )
            elif 'client_id' in cred_data and 'refresh_token' in cred_data:
                _safe_print("   Detected user account credentials")
                credentials = Credentials.from_authorized_user_file(self.credentials_path)
                if credentials.expired:
                    _safe_print("   Refreshing expired credentials...")
                    credentials.refresh(requests.Request())
            else:
                _safe_print(f"   Unknown credentials format; keys: {list(cred_data.keys())}")
                _safe_print("   Falling back to application default")
                credentials, _ = default()
            _safe_print("   Creating BigQuery client...")
            client = bigquery.Client(credentials=credentials, project=self.project_id)
            _safe_print("   Client created successfully")
            return client
        except Exception as e:
            self._log_error("_create_client_with_credentials", e, f"credentials_path={self.credentials_path}")
            _safe_print("   Falling back to default credentials")
            return bigquery.Client(project=self.project_id)

    def query(self, sql: str) -> Optional[pd.DataFrame]:
        try:
            _safe_print("Executing query...")
            _safe_print(f"   Query preview: {sql[:100]}{'...' if len(sql) > 100 else ''}")
            query_job = self.client.query(sql)
            _safe_print("   Waiting for results...")
            results = query_job.result()
            _safe_print("   Converting to DataFrame...")
            df = results.to_dataframe(create_bqstorage_client=False)
            _safe_print(f"Query executed. Rows: {len(df):,}")
            return df
        except Exception as e:
            lowered = str(e).lower()
            if "db-dtypes" in lowered:
                _safe_print("Missing package db-dtypes. Install: pip install db-dtypes")
            elif "pyarrow" in lowered:
                _safe_print("Missing package pyarrow. Install: pip install pyarrow")
            else:
                self._log_error("query", e, f"SQL length={len(sql)}")
            return None

    def query_to_csv(self, sql: str, filename: str, add_timestamp: bool = True) -> Optional[str]:
        df = self.query(sql)
        if df is not None:
            return self.save_dataframe(df, filename, add_timestamp)
        return None

    def save_to_table(self, df: pd.DataFrame, dataset_id: str, table_id: str,
                      write_disposition: str = 'WRITE_APPEND', create_if_not_exists: bool = True) -> bool:
        try:
            _safe_print("Saving DataFrame to BigQuery table...")
            _safe_print(f"   Target: {self.project_id}.{dataset_id}.{table_id}")
            _safe_print(f"   Shape: {df.shape}")
            table_exists = self.table_exists(dataset_id, table_id)
            _safe_print(f"   Table exists: {table_exists}")
            if not table_exists and not create_if_not_exists:
                _safe_print("Table absent and create_if_not_exists is False")
                return False
            table_ref = self.client.dataset(dataset_id).table(table_id)
            job_config = bigquery.LoadJobConfig(write_disposition=write_disposition,
                                                autodetect=False if table_exists else True)
            _safe_print("   Starting load job...")
            job = self.client.load_table_from_dataframe(df, table_ref, job_config=job_config)
            _safe_print("   Waiting for completion...")
            job.result()
            _safe_print(f"Saved {len(df):,} rows to {dataset_id}.{table_id}")
            info = self.get_table_info(dataset_id, table_id)
            if info:
                _safe_print(f"   Table now contains {info['num_rows']:,} rows")
            return True
        except Exception as e:
            self._log_error("save_to_table", e, f"dataset_id={dataset_id}, table_id={table_id}, shape={df.shape}")
            return False

    def get_table_info(self, dataset_id: str, table_id: str) -> Optional[Dict[str, Any]]:
        try:
            _safe_print(f"Getting table info: {dataset_id}.{table_id}")
            table_ref = self.client.dataset(dataset_id).table(table_id)
            table = self.client.get_table(table_ref)
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
                        'name': f.name,
                        'type': f.field_type,
                        'mode': f.mode,
                        'description': f.description
                    } for f in table.schema
                ]
            }
            _safe_print("Table info retrieved")
            return info
        except Exception as e:
            self._log_error("get_table_info", e, f"dataset_id={dataset_id}, table_id={table_id}")
            return None

    def table_exists(self, dataset_id: str, table_id: str) -> bool:
        try:
            table_ref = self.client.dataset(dataset_id).table(table_id)
            self.client.get_table(table_ref)
            _safe_print(f"Table exists: {dataset_id}.{table_id}")
            return True
        except Exception:
            _safe_print(f"Table not found or inaccessible: {dataset_id}.{table_id}")
            return False

    def list_datasets(self) -> List[str]:
        try:
            datasets = list(self.client.list_datasets())
            ids = [d.dataset_id for d in datasets]
            _safe_print(f"Datasets ({len(ids)}): {ids}")
            return ids
        except Exception as e:
            self._log_error("list_datasets", e, f"project_id={self.project_id}")
            return []

    def list_tables(self, dataset_id: str) -> List[str]:
        try:
            dataset_ref = self.client.dataset(dataset_id)
            tables = list(self.client.list_tables(dataset_ref))
            ids = [t.table_id for t in tables]
            _safe_print(f"Tables in {dataset_id} ({len(ids)}): {ids}")
            return ids
        except Exception as e:
            self._log_error("list_tables", e, f"dataset_id={dataset_id}")
            return []

    def get_schema(self, dataset_id: str, table_id: str) -> Optional[List[Dict[str, str]]]:
        info = self.get_table_info(dataset_id, table_id)
        return info['schema'] if info else None

    def sample_table(self, dataset_id: str, table_id: str, limit: int = 5) -> Optional[pd.DataFrame]:
        try:
            sql = f"SELECT * FROM `{self.project_id}.{dataset_id}.{table_id}` LIMIT {limit}"
            return self.query(sql)
        except Exception as e:
            self._log_error("sample_table", e, f"dataset_id={dataset_id}, table_id={table_id}, limit={limit}")
            return None

    @staticmethod
    def save_dataframe(df: pd.DataFrame, filename: str, add_timestamp: bool = True) -> str:
        try:
            _safe_print("Saving DataFrame to CSV...")
            _safe_print(f"   Shape: {df.shape}")
            if add_timestamp:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out = f"{filename}_{ts}.csv"
            else:
                out = f"{filename}.csv"
            _safe_print(f"   Output file: {out}")
            df.to_csv(out, index=False)
            _safe_print(f"Data saved to {out}")
            return out
        except Exception as e:
            _safe_print(f"Error saving DataFrame: {e}")
            raise

    def _log_error(self, operation: str, error: Exception, additional_context: str = "") -> None:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        _safe_print(f"Error in {operation}:")
        _safe_print(f"   Type: {type(error).__name__}")
        _safe_print(f"   Message: {error}")
        if additional_context:
            _safe_print(f"   Context: {additional_context}")
        if exc_traceback:
            tb = exc_traceback
            while tb.tb_next is not None:
                tb = tb.tb_next
            filename = tb.tb_frame.f_code.co_filename
            line_number = tb.tb_lineno
            func = tb.tb_frame.f_code.co_name
            if 'bigquery_connector.py' in filename:
                _safe_print(f"   Location: {func} line {line_number}")
        if os.getenv('DEBUG_MODE', '').lower() in ['true', '1', 'yes']:
            _safe_print("   Traceback:")
            traceback.print_exc()
        _safe_print("")

    def __repr__(self) -> str:
        return f"BigQueryConnector(project_id='{self.project_id}')"

def quick_query(sql: str, project_id: Optional[str] = None) -> Optional[pd.DataFrame]:
    try:
        _safe_print("Quick query execution...")
        bq = BigQueryConnector(project_id=project_id)
        return bq.query(sql)
    except Exception as e:
        _safe_print(f"Error in quick_query: {e}")
        return None

def get_kramp_purchase_data(limit: int = 1000) -> Optional[pd.DataFrame]:
    try:
        _safe_print(f"Getting Kramp purchase data (limit={limit})...")
        sql = f"""
        SELECT * FROM `kramp-sharedmasterdata-prd.MadsH.purchase_data`
        WHERE LOWER(TRIM(class4)) IN (
            '6905 | bolts & nuts 8.8 metric',
            '6910 | bolts & nuts 10.9 metric',
            '6935 | bolts & nuts stainless steel',
            '6965 | washers',
            '6952 | threaded rods 8.8 - 10.9',
            '6900 | bolts & nuts 4.6 metric',
            '6915 | bolts & nuts 12.9 metric',
            '6920 | bolts & nuts metric fine',
            '6945 | bolts & nuts other',
            '6970 | washers stainless steel',
            '6944 | metal screws',
            '6925 | bolts & nuts unc / unf',
            '6954 | threaded rods stainless steel',
            '6985 | wood screws',
            '7008 | shims',
            '6930 | bolts & nuts hotdip galvanized',
            '6950 | threaded rods 4.6',
            '6981 | wall fixings stainless steel',
            '6956 | threaded rods trapizoidal',
            '6984 | wall fixings other'
        )
        LIMIT {limit}
        """
        return quick_query(sql)
    except Exception as e:
        _safe_print(f"Error in get_kramp_purchase_data: {e}")
        return None


