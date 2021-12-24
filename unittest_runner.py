import unittest

loader = unittest.TestLoader()
start_path = '/Users/georgiosspyrou/Desktop/GitHub/pydistr'

suite = loader.discover(start_path)

runner = unittest.TextTestRunner()
runner.run(suite)
