

import unittest
from GenericDistribution import ProbabilityDistribution

generic_test_cases_file = '/Users/georgiosspyrou/Desktop/GitHub/pydistr/input_files/normal_data.txt'


class TestReadDataFile(unittest.TestCase):

    def testCalculateMean(self):
        pds = ProbabilityDistribution()
        pds.read_data_from_text_file(filename=generic_test_cases_file)
        self.assertEqual(sum(pds.data), 1170.0, msg='Counts do not match')
