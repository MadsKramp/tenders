from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from .bq import run_sql

# Friendly display names
DISPLAY_NAME: Dict[str, str] = {
    "crm_main_group_vendor": "Group Vendor",
    "crm_main_vendor": "Main Vendor",
    "purchase_amount_eur": "Purchase Amount (EUR)",
    "purchase_quantity": "Purchase Quantity",
}

# ---------- Top-N vendors per filtered period, returned by year ----------
def vendor_topN_by_year_sql(full_table: str, year_start: int, year_end: int, limit: int = 20) -> Tuple[str, Dict[str, Any]]:
    sql = f"""
    WITH base AS (
      SELECT
        CAST(year_authorization AS INT64) AS year_authorization,
        crm_main_group_vendor,
        CAST(purchase_amount_eur AS NUMERIC) AS purchase_amount_eur
      FROM `{full_table}`
      WHERE purchase_amount_eur IS NOT NULL
        AND year_authorization BETWEEN @year_start AND @year_end
    ),
    agg AS (
      SELECT
        crm_main_group_vendor,
        year_authorization,
        SUM(purchase_amount_eur) AS total_purchase_eur,
        COUNT(1) AS txn_count
      FROM base
      GROUP BY 1,2
    ),
    totals AS (
      SELECT
        crm_main_group_vendor,
        SUM(total_purchase_eur) AS total_over_period
      FROM agg
      GROUP BY 1
    ),
    topN AS (
      SELECT crm_main_group_vendor, total_over_period
      FROM totals
      ORDER BY total_over_period DESC
      LIMIT {int(limit)}
    )
    SELECT
      a.crm_main_group_vendor,
      a.year_authorization,
      a.total_purchase_eur,
      a.txn_count,
      t.total_over_period
    FROM agg a
    JOIN topN t
    USING (crm_main_group_vendor)
    ORDER BY t.total_over_period DESC, a.crm_main_group_vendor, a.year_authorization
    """
    params = {"year_start": int(year_start), "year_end": int(year_end)}
    return sql, params

def topN_by_year(full_table: str, year_start: int, year_end: int, limit: int = 20) -> pd.DataFrame:
    """Return Top-N (default 20) Group Vendors over [year_start, year_end], with yearly rows."""
    sql, params = vendor_topN_by_year_sql(full_table, year_start, year_end, limit)
    df = run_sql(sql, params)
    # Types
    df["total_purchase_eur"] = pd.to_numeric(df["total_purchase_eur"], errors="coerce").astype("float64")
    df["txn_count"]          = pd.to_numeric(df["txn_count"], errors="coerce").astype("int64")
    df["year_authorization"] = pd.to_numeric(df["year_authorization"], errors="coerce").astype("int64")
    df["total_over_period"]  = pd.to_numeric(df["total_over_period"], errors="coerce").astype("float64")
    # Friendly names
    df = df.rename(columns={
        "crm_main_group_vendor": DISPLAY_NAME["crm_main_group_vendor"],
        "year_authorization": "Year",
        "total_purchase_eur": "Total Purchase (EUR)",
        "txn_count": "Transactions",
        "total_over_period": "Total Over Period (EUR)",
    })
    return df

  # Backward-compatible alias
  top20_by_year = topN_by_year

def overview_topN(df: pd.DataFrame) -> pd.DataFrame:
    """Sorted overview of Top-N vendors for the whole period."""
    out = (df[["Group Vendor","Total Over Period (EUR)"]]
            .drop_duplicates()
            .sort_values("Total Over Period (EUR)", ascending=False))
    return out

def pivot_topN_yearly(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to Group Vendor x Year with yearly spend, ordered by period total (desc) and years asc."""
    order = (df[["Group Vendor","Total Over Period (EUR)"]]
             .drop_duplicates()
             .sort_values("Total Over Period (EUR)", ascending=False)["Group Vendor"].tolist())
    pivot = (df.pivot_table(index="Group Vendor", columns="Year",
                            values="Total Purchase (EUR)", aggfunc="sum")
               .fillna(0.0)
               .reindex(order))
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    return pivot

# ---------- Formatting & plotting helpers ----------
def fmt_eur(x: float) -> str:
    """Format numeric to '1,234,567 €' with no decimals."""
    if pd.isna(x):
        return ""
    return f"{x:,.0f} €"

def plot_scatter_eur(df: pd.DataFrame, *, title: str | None = None, logy: bool = False) -> None:
    """Scatter of (Year vs Total Purchase (EUR)) for the given Top-N dataframe."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    fig, ax = plt.subplots()
    ax.scatter(df["Year"], df["Total Purchase (EUR)"])
    ax.set_title(title or "Top Group Vendors — Yearly Spend (EUR)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Purchase (EUR)")
    if logy:
        ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:,.0f} €"))
    plt.show()
