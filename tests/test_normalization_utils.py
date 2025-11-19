import unittest
import pandas as pd
from source.data_processing.normalization_utils import (
    resolve_spend,
    resolve_class3,
    resolve_product_identifiers,
    resolve_all,
)


class TestNormalizationUtils(unittest.TestCase):
    def test_resolve_spend_direct(self):
        df = pd.DataFrame({'purchase_amount_eur':[100,200],'purchase_quantity':[1,2]})
        out, col = resolve_spend(df, spend_col='total_spend')
        self.assertEqual(col, 'total_spend')
        # Logic chooses direct purchase_amount_eur column; expect simple sum 100+200=300
        self.assertEqual(out[col].sum(), 300)

    def test_resolve_class3_generic(self):
        df = pd.DataFrame({'Class':['A','B'], 'Other':['x','y']})
        out, col = resolve_class3(df)
        self.assertEqual(col, 'Class3')
        self.assertIn('Class3', out.columns)

    def test_resolve_identifiers_synthesis(self):
        df = pd.DataFrame({'val':[1,2,3]})
        out, pn, pd_, pq = resolve_product_identifiers(df)
        self.assertEqual(pn, 'ProductNumber')
        self.assertEqual(pd_, 'ProductDescription')
        self.assertEqual(pq, 'PurchaseQuantity')
        self.assertEqual(out[pq].sum(), 0)

    def test_resolve_all_pipeline(self):
        df = pd.DataFrame({'purchase_amount_eur':[10,20], 'purchase_quantity':[2,3], 'Class':['C','D']})
        out = resolve_all(df)
        self.assertIn('total_spend', out.columns)
        self.assertIn('Class3', out.columns)
        self.assertIn('ProductNumber', out.columns)


if __name__ == '__main__':
    unittest.main()
