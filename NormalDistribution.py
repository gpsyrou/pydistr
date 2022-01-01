
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

    def get_population_size(self, is_sample: bool = True) -> int:
        """ Method to retrieve the sample or population size N.
        """
        n = len(self.data)
        if is_sample:
            n = n - 1

        return n

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
        n = self.get_population_size(is_sample=is_sample)
        self.variance = (sum([(xi - mu) ** 2 for xi in self.data])) / n

        return self.variance

    def calculate_std(self, is_sample=True):
        """ Method that computes the standard deviation of the normal
        distribution based on the data attribute.
        """
        var = self.calculate_variance(is_sample=is_sample)
        self.stdev = np.sqrt(var)

        return self.stdev

    def compute_pdf(self, x: float) -> float:
        """ Calculate the probability density function for the normal
        distribution, based on the following formula:
             (1/s*sqrt(2*pi)) * exp(-(x-m / s)^2) / 2)

        Args
        ----
            x: Point for calculating the pdf

        Returns
        -------
            pdf: Probability density function (PDF) for point x
        """
        calc_one = 1.0 / (self.stdev * (math.sqrt(2 * math.pi)))
        calc_two = math.exp((-0.5*((x - self.mean) / self.stdev) ** 2))
        pdf = calc_one * calc_two

        return pdf

    def plot_histogram(self, n_bins: int = 10):
        """ Output a histogram of the data attribute.
        """
        ax = sns.distplot(a=self.data, bins=n_bins, norm_hist=False)
        ax.set_title('Histogram of Data')
        ax.set_xlabel('Data')
        ax.set_ylabel('Count')
        plt.show()

    def __repr__(self):
        """ Method to return the characteristics of a Gaussian distribution
        instance. Currently the method outputs:
            1) mean
            2) variance
            3) standard deviation
        """
        return 'Mean: {0} \nVariance: {1} \nStandard Deviation: {2}'.format(self.mean, self.variance, self.stdev)
