from __future__ import annotations
from typing import Any, Dict, Optional
import time, logging, os
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPICallError, RetryError
import pandas as pd

_LOG = logging.getLogger(__name__)


def make_client() -> bigquery.Client:
    return bigquery.Client(
        project=os.getenv("kramp-sharedmasterdata-prd.MadsH.purchase_data"),
        location=os.getenv("BQ_LOCATION", "EU"),
    )


def _bq_type(v: Any) -> str:
    if isinstance(v, bool):
        return "BOOL"
    if isinstance(v, int):
        return "INT64"
    if isinstance(v, float):
        return "FLOAT64"
    return "STRING"


def run_sql(
    query: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    max_retries: int = 3,
    timeout: int = 1800,
) -> pd.DataFrame:
    client = make_client()
    job_config = None
    if params:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(k, _bq_type(v), v) for k, v in params.items()
            ]
        )
    for attempt in range(1, max_retries + 1):
        try:
            job = client.query(query, job_config=job_config)
            return job.result(timeout=timeout).to_dataframe(create_bqstorage_client=True)
        except (GoogleAPICallError, RetryError, TimeoutError) as e:
            _LOG.warning("BQ attempt %s failed: %s", attempt, e)
            if attempt == max_retries:
                raise
            time.sleep(2 * attempt)
