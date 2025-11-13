from __future__ import annotations
import pandas as pd
import numpy as np

# Requires: ProductNumber, purchase_amount_eur, purchase_quantity

def product_agg_for_clustering(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("ProductNumber", as_index=False)
        .agg(
            total_purchase_eur=("purchase_amount_eur", "sum"),
            total_quantity=("purchase_quantity", "sum"),
            txn_count=("purchase_amount_eur", "size"),
        )
    )
    g["log_total_purchase_eur"] = np.log1p(g["total_purchase_eur"])
    return g[
        [
            "ProductNumber",
            "total_purchase_eur",
            "total_quantity",
            "txn_count",
            "log_total_purchase_eur",
        ]
    ]
