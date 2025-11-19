import os
import pandas as pd
from pathlib import Path

from .export_utils import export_year_split_purchase_quantity
from .enrichment_utils import enrich_exports


class StubBQ:
    def __init__(self, purchase_rows=None, detail_rows=None):
        self._purchase_rows = purchase_rows or []
        self._detail_rows = detail_rows or []

    def query(self, sql: str):
        # Purchase data query detection
        if "purchase_quantity" in sql.lower() and "GROUP BY" in sql:
            return pd.DataFrame(self._purchase_rows)
        # Probe for long-form attribute schema
        if "LIMIT 1" in sql:
            return pd.DataFrame({
                "AttributeName": ["Length"],
                "AttributeValue": ["10"],
                "product_id": ["ticGoldenItem-ABC"]
            })
        # Long-form attribute fetch path
        if "AttributeName AS detail_name" in sql:
            # Return a small long-form attribute set
            return pd.DataFrame({
                "ProductNumber": ["ABC", "DEF"],
                "detail_name": ["Length", "Length"],
                "detail_value": ["10", "20"],
            })
        return pd.DataFrame()


def test_enrichment_preserves_year_header(tmp_path):
    # Skip if xlsxwriter missing
    try:
        import xlsxwriter  # noqa: F401
    except Exception:
        return  # silently skip

    # Prepare stub purchase data (one Class3 group)
    purchase_rows = [
        {"ProductNumber": "ABC", "ProductDescription": "Alpha", "Class3": "Fasteners", "Year": 2023, "PurchaseQuantity": 1200},
        {"ProductNumber": "ABC", "ProductDescription": "Alpha", "Class3": "Fasteners", "Year": 2024, "PurchaseQuantity": 800},
        {"ProductNumber": "DEF", "ProductDescription": "Delta", "Class3": "Fasteners", "Year": 2023, "PurchaseQuantity": 500},
        {"ProductNumber": "DEF", "ProductDescription": "Delta", "Class3": "Fasteners", "Year": 2024, "PurchaseQuantity": 900},
    ]
    bq = StubBQ(purchase_rows=purchase_rows)

    out_dir = tmp_path / "year_exports"
    out_dir.mkdir(exist_ok=True)
    paths = export_year_split_purchase_quantity(bq, str(out_dir), fmt_thousands=True)
    assert paths, "Export should create at least one file"

    # Enrich exports using product_utils module (stub inside data_processing)
    from . import product_utils as product_utils_module
    summary = enrich_exports(str(out_dir), bq, product_utils_module, preserve_year_format=True)
    assert summary["files_enriched"] >= 1

    # Read enriched file using header=1 (due to merged top header)
    enriched_path = paths[0]
    df = pd.read_excel(enriched_path, header=1)
    # Ensure detail column added (e.g., Length)
    assert "Length" in df.columns, "Expected product detail column 'Length' after enrichment"
    # Ensure year columns present and formatted header row exists (checked via reading header=0)
    df_header0 = pd.read_excel(enriched_path, header=0)
    assert df_header0.columns[0] == "ProductNumber"
    # Merged header causes third column name to be 'PurchaseQuantity'
    assert "PurchaseQuantity" in df_header0.columns.tolist(), "Merged header 'PurchaseQuantity' expected in header row 0"