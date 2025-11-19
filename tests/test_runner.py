import unittest
import sys
import os

if __name__ == '__main__':
    root = os.path.dirname(__file__)
    suite = unittest.defaultTestLoader.discover(root, pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
