from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd

CACHE_DIR = Path("data/.cache"); CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(sql: str, params: dict | None) -> str:
    s = sql + "|" + json.dumps(params or {}, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def read_or_query(sql: str, params: dict | None, fetch_fn, *, force: bool = False) -> pd.DataFrame:
    key = _cache_key(sql, params)
    fp = CACHE_DIR / f"{key}.parquet"
    if fp.exists() and not force:
        return pd.read_parquet(fp)
    df = fetch_fn(sql, params)
    df.to_parquet(fp, index=False)
    return df
