
import unittest
import numpy as np
from GenericDistribution import ProbabilityDistribution
from NormalDistribution import NormalDistribution

generic_test_cases_file = '/Users/georgiosspyrou/Desktop/GitHub/pydistr/input_files/normal_data.txt'


# Read File Unit Tests
class TestReadDataFile(unittest.TestCase):

    def testDataFileReadCorrectly(self):
        pds = ProbabilityDistribution()
        pds.read_data_from_text_file(filename=generic_test_cases_file)
        self.assertEqual(np.round(np.sum(pds.data), 3), 2.88, msg='Counts do not match')


# Normal Distribution Unit Tests
class TestNormalDistribution(unittest.TestCase):
    def testNormalDistrMeanCalculation(self):
        pds = NormalDistribution()
        pds.read_data_from_text_file(filename=generic_test_cases_file)
        mu = pds.calculate_mean()
        mu = np.round(mu, 4)
        self.assertEqual(mu, 0.0029, msg='The mean does not compute correctly')

    def testNormalDistrVarianceCalculation(self):
        pds = NormalDistribution()
        pds.read_data_from_text_file(filename=generic_test_cases_file)
        var_pop = pds.calculate_variance(is_sample=False)
        var_sample = pds.calculate_variance(is_sample=True)
        self.assertEqual(
            np.round(var_pop, 4),
            0.9988,
            msg='The variance for population is not computed correctly'
        )
        self.assertEqual(
            np.round(var_sample, 4),
            0.9998,
            msg='The variance for sample is not computed correctly'
        )

    def testNormalDistrStdCalculation(self):
        pds = NormalDistribution()
        pds.read_data_from_text_file(filename=generic_test_cases_file)
        std_pop = pds.calculate_std(is_sample=False)
        std_sample = pds.calculate_std(is_sample=True)
        self.assertEqual(
            np.round(std_pop, 4),
            0.9994,
            msg='The stdv for population is not computed correctly'
        )
        self.assertEqual(
            np.round(std_sample, 4),
            0.9999,
            msg='The stdv for sample is not computed correctly'
        )

    def testPDFPointCalculation(self):
        pds = NormalDistribution()
        pds.read_data_from_text_file(filename=generic_test_cases_file)
        pds.calculate_mean()
        pds.calculate_std()
        val = pds.compute_pdf(10)
        self.assertEqual(
            np.round(val*(10**19), 6),
            0.000784,
            msg='PDF point calculation is not computed correctly'
        )
