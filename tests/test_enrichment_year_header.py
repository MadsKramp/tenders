import unittest
import pandas as pd
from source.data_processing.export_utils import export_year_split_purchase_quantity
from source.data_processing.enrichment_utils import enrich_exports


class StubBQ:
    def __init__(self, purchase_rows=None):
        self._purchase_rows = purchase_rows or []

    def query(self, sql: str):
        low = sql.lower()
        if "purchase_quantity" in low and "group by" in low:
            return pd.DataFrame(self._purchase_rows)
        if "limit 1" in low:
            return pd.DataFrame({
                "AttributeName": ["Length"],
                "AttributeValue": ["10"],
                "product_id": ["ticGoldenItem-ABC"]
            })
        if "attributevalue" in low and "detail_name" in low:
            return pd.DataFrame({
                "ProductNumber": ["ABC", "DEF"],
                "detail_name": ["Length", "Length"],
                "detail_value": ["10", "20"],
            })
        return pd.DataFrame()


class TestEnrichmentYearHeader(unittest.TestCase):
    def test_enrichment_preserves_year_header(self):
        try:
            import xlsxwriter  # noqa: F401
        except Exception:
            self.skipTest("xlsxwriter not available")

        purchase_rows = [
            {"ProductNumber": "ABC", "ProductDescription": "Alpha", "Class3": "Fasteners", "Year": 2023, "PurchaseQuantity": 1200},
            {"ProductNumber": "ABC", "ProductDescription": "Alpha", "Class3": "Fasteners", "Year": 2024, "PurchaseQuantity": 800},
            {"ProductNumber": "DEF", "ProductDescription": "Delta", "Class3": "Fasteners", "Year": 2023, "PurchaseQuantity": 500},
            {"ProductNumber": "DEF", "ProductDescription": "Delta", "Class3": "Fasteners", "Year": 2024, "PurchaseQuantity": 900},
        ]
        bq = StubBQ(purchase_rows=purchase_rows)

        import tempfile, pathlib
        tmp_dir = pathlib.Path(tempfile.mkdtemp())
        out_dir = tmp_dir / "year_exports"
        out_dir.mkdir(exist_ok=True)
        paths = export_year_split_purchase_quantity(bq, str(out_dir), fmt_thousands=True)
        self.assertTrue(paths, "Export should create at least one file")

        from source.data_processing import product_utils as product_utils_module
        summary = enrich_exports(str(out_dir), bq, product_utils_module, preserve_year_format=True)
        self.assertGreaterEqual(summary["files_enriched"], 1)

        enriched_path = paths[0]
        df = pd.read_excel(enriched_path, header=1)
        self.assertIn("Length", df.columns, "Expected product detail column 'Length' after enrichment")

        df_header0 = pd.read_excel(enriched_path, header=0)
        self.assertEqual(df_header0.columns[0], "ProductNumber")
        self.assertIn("PurchaseQuantity", list(df_header0.columns), "Merged header 'PurchaseQuantity' expected in header row 0")


if __name__ == "__main__":
    unittest.main()
