
import numpy as np
from GenericDistribution import ProbabilityDistribution


class NormalDistribution(ProbabilityDistribution):
    """ Distribution class for calculating and visualizing a Gaussian 
    probability distribution.
    """
    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        """ The generic distribution has mean of 0 and variance/std of 1,
        indicating a standard normal distribution.
        """
        ProbabilityDistribution.__init__(self, mu, sigma)

    def calculate_mean(self):
        """ Method that computes the mean of the normal distribution based
        on the data attribute.
        The computation is using the following formula:
            mu = sum(xi) / N, where xi are the data points in data attribute
        """
        self.mean = sum(self.data) / len(self.data)
        return self.mean

    def calculate_variance(self, is_sample=True):
        """ Method that computes the variance of the normal distribution based
        on the data attribute.

        The computation is using the following formula:
            variance = sum(xi-mu) / N-1 (sample) or sum(xi-mu) / N (population)
            where
                xi: are the data points in data attribute
                mu: is the mean of the distribution
        """
        mu = self.calculate_mean()

        n = len(self.data)
        if is_sample:
            n = n - 1

        self.variance = (sum([(xi - mu) ** 2 for xi in self.data])) / n

        return self.variance

    def calculate_std(self, is_sample=True):
        """ Method that computes the standard deviation of the normal 
        distribution based on the data attribute.
        """
        n = len(self.data)
        if is_sample:
            n = n - 1
        var = self.calculate_variance(is_sample=is_sample)
        self.stdev = np.sqrt(var)
        return self.stdev
