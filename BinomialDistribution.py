
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from GenericDistribution import ProbabilityDistribution


class BinomialDistribution(ProbabilityDistribution):
    """ Binomial distribution class the performs computations
    and visualization for a binomial distribution.
    Attributes:
    -----------
        n: number of trials
        p: probability of an event occuring
        mean: describes the mean value of the distribution
        variance: describes the variance of the distribution
        stdev: described the standard deviation of the distribution
    """
    def __init__(self, prob: float = 0.5, n_trials: int = 20):
        self.p = prob
        self.n = n_trials

        ProbabilityDistribution.__init__(self, mu=self.calculate_mean(), sigma=self.calculate_std())

    def calculate_mean(self):
        """ Compute the mean for a binomial distribution:
            mu = n * p
        """
        self.mean = self.n * self.p

        return self.mean

    def calculate_variance(self):
        """ Compute the variance for a binomial distribution:
            variance = n * p * (1-p)
        """
        self.variance = self.n * self.p * (1-self.p)

        return self.variance

    def calculate_std(self):
        """ Compute the variance for a binomial distribution:
            stdev = sqrt(n * p * (1-p))
        """
        variance = self.calculate_variance()
        self.stdev = np.sqrt(variance)

        return self.stdev
