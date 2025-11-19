import unittest
import tempfile
import os
import pandas as pd
from source.data_processing.export_utils import export_year_split_purchase_quantity
from tests.conftest import MockBQ


class TestExportUtils(unittest.TestCase):
    def test_export_year_split_purchase_quantity(self):
        df = pd.DataFrame({
            'ProductNumber':['P1','P1','P2','P2'],
            'ProductDescription':['Desc1','Desc1','Desc2','Desc2'],
            'Class3':['C1','C1','C2','C2'],
            'Year':[2023,2024,2023,2024],
            'PurchaseQuantity':[1000,2500,300,700]
        })
        bq = MockBQ(df)
        tmp = tempfile.mkdtemp(prefix='export_test_')
        try:
            paths = export_year_split_purchase_quantity(bq, tmp, table='ignored', fmt_thousands=False)
            self.assertEqual(len(paths), 2)
            for p in paths:
                self.assertTrue(os.path.isfile(p))
                read = pd.read_excel(p)
                year_cols = {'2023','2024'}
                self.assertTrue(year_cols.issubset(set(read.columns)))
                self.assertIn(read['2023'].dtype.kind, {'i','u'})
        finally:
            for p in paths:
                if os.path.exists(p):
                    os.remove(p)
            os.rmdir(tmp)


if __name__ == '__main__':
    unittest.main()
