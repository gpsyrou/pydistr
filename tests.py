
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
        self.assertEqual(sum(pds.data), 1170.0, msg='Counts do not match')


# Normal Distribution Unit Tests
class TestNormalDistribution(unittest.TestCase):
    def testNormalDistrMeanCalculation(self):
        pds = NormalDistribution()
        pds.read_data_from_text_file(filename=generic_test_cases_file)
        mu = pds.calculate_mean()
        self.assertEqual(mu, 58.5, msg='The mean does not compute correctly')

    def testNormalDistrVarianceCalculation(self):
        pds = NormalDistribution()
        pds.read_data_from_text_file(filename=generic_test_cases_file)
        var_pop = pds.calculate_variance(is_sample=False)
        var_sample = pds.calculate_variance(is_sample=True)
        self.assertEqual(
            np.round(var_pop, 3),
            667.05,
            msg='The variance for population is not computed correctly'
        )
        self.assertEqual(
            np.round(var_sample, 3),
            702.158,
            msg='The variance for sample is not computed correctly'
        )

    def testNormalDistrStdCalculation(self):
        pds = NormalDistribution()
        pds.read_data_from_text_file(filename=generic_test_cases_file)
        std_pop = pds.calculate_std(is_sample=False)
        std_sample = pds.calculate_std(is_sample=True)
        self.assertEqual(
            np.round(std_pop, 3),
            25.827,
            msg='The stdv for population is not computed correctly'
        )
        self.assertEqual(
            np.round(std_sample, 3),
            26.498,
            msg='The stdv for sample is not computed correctly'
        )

    def testPDFPointCalculation(self):
        pds = NormalDistribution()
        pds.read_data_from_text_file(filename=generic_test_cases_file)
        pds.calculate_mean()
        pds.calculate_std()
        self.assertEqual(
            np.round(pds.compute_pdf(10), 4),
            0.0028,
            msg='PDF point calculation is not computed correctly'
        )
