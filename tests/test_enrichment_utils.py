import unittest
import os
import pandas as pd
from tests.conftest import (
    make_mock_product_utils_wide,
    make_temp_export_dir,
    write_excel,
    cleanup_dir,
    MockBQ,
)
from source.data_processing.enrichment_utils import enrich_exports


class TestEnrichmentUtils(unittest.TestCase):
    def test_enrich_exports_wide(self):
        export_dir = make_temp_export_dir()
        try:
            df1 = pd.DataFrame({'ProductNumber':['P1','P2'], 'ProductDescription':['D1','D2']})
            df2 = pd.DataFrame({'ProductNumber':['P3'], 'ProductDescription':['D3']})
            write_excel(os.path.join(export_dir,'file1.xlsx'), df1)
            write_excel(os.path.join(export_dir,'file2.xlsx'), df2)
            details_df = pd.DataFrame({
                'ProductNumber':['P1','P2','P3'],
                'AttrA':['X','Y','Z'],
                'AttrB':[10,20,30]
            })
            mock_mod = make_mock_product_utils_wide()
            def get_product_details_mapping(*args, **kwargs):
                return details_df.copy()
            mock_mod.get_product_details_mapping = get_product_details_mapping
            bq = MockBQ(pd.DataFrame())  # not used by mock
            summary = enrich_exports(export_dir, bq, mock_mod)
            self.assertEqual(summary['files_enriched'], 2)
            f1 = pd.read_excel(os.path.join(export_dir,'file1.xlsx'))
            self.assertIn('AttrA', f1.columns)
            self.assertIn('AttrB', f1.columns)
        finally:
            cleanup_dir(export_dir)


if __name__ == '__main__':
    unittest.main()
