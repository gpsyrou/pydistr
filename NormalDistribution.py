
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

    def compute_mean(self):
        """ Method that computes the mean of the distribution based on the
        data attribute.
        mean = sum(xi) / N
        """
        self.mean = sum(self.data) / len(self.data)
        return self.mean
