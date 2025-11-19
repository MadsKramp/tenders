import pandas as pd
import tempfile
import os
import shutil
import types

class MockBQ:
    def __init__(self, df):
        self._df = df
    def query(self, q):
        return self._df.copy()

class MockProductUtilsModule(types.SimpleNamespace):
    pass

def make_mock_product_utils_wide():
    df = pd.DataFrame({
        'ProductNumber': ['P1','P2','P3'],
        'AttrA': ['X','Y','Z'],
        'AttrB': [10, 20, 30]
    })
    mod = MockProductUtilsModule()
    def get_product_details_mapping(*args, **kwargs):
        return df.copy()
    mod.get_product_details_mapping = get_product_details_mapping
    return mod

def make_temp_export_dir():
    tmp = tempfile.mkdtemp(prefix='exports_')
    return tmp

def write_excel(path, df):
    df.to_excel(path, index=False)

def cleanup_dir(d):
    if os.path.isdir(d):
        shutil.rmtree(d)
