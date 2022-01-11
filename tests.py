
import unittest
import numpy as np
from GenericDistribution import ProbabilityDistribution
from NormalDistribution import NormalDistribution
from BinomialDistribution import BinomialDistribution

normal_tests_file = '/Users/georgiosspyrou/Desktop/GitHub/pydistr/input_files/normal_data.txt'
binomial_tests_file = '/Users/georgiosspyrou/Desktop/GitHub/pydistr/input_files/binomial_data.txt'


class TestReadDataFile(unittest.TestCase):

    def testDataFileReadCorrectly(self):
        pds = ProbabilityDistribution()
        pds.read_data_from_text_file(filename=normal_tests_file)
        self.assertEqual(
            np.round(np.sum(pds.data), 3),
            2.88,
            msg='Counts do not match'
            )


# Normal Distribution Unit Tests
class TestNormalDistribution(unittest.TestCase):
    def testNormalDistrMeanCalculation(self):
        pds = NormalDistribution()
        pds.read_data_from_text_file(filename=normal_tests_file)
        mu = pds.calculate_mean()
        mu = np.round(mu, 4)
        self.assertEqual(mu, 0.0029, msg='The mean does not compute correctly')

    def testNormalDistrVarianceCalculation(self):
        pds = NormalDistribution()
        pds.read_data_from_text_file(filename=normal_tests_file)
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
        pds.read_data_from_text_file(filename=normal_tests_file)
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
        pds.read_data_from_text_file(filename=normal_tests_file)
        pds.calculate_mean()
        pds.calculate_std()
        val = pds.compute_pdf(10)
        self.assertEqual(
            np.round(val*(10**19), 6),
            0.000784,
            msg='PDF point calculation is not computed correctly'
        )


class TestBinomialDistribution(unittest.TestCase):
    def testBinomialDistrMeanCalculation(self):
        pds = BinomialDistribution()
        mu = pds.calculate_mean()
        self.assertEqual(mu, 10.0, msg='The mean does not compute correctly')

    def testBinomialDistrVarianceCalculation(self):
        pds = BinomialDistribution()
        var = pds.calculate_variance()
        self.assertEqual(
            var,
            5.0,
            msg='The variance is not computed correctly'
            )

    def testBinomialDistrStdCalculation(self):
        pds = BinomialDistribution()
        std = pds.calculate_std()
        std = np.round(std, 4)
        self.assertEqual(
            std,
            2.2361,
            msg='The standard deviation is not computed correctly'
            )
