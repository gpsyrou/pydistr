
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from GenericDistribution import ProbabilityDistribution


class NormalDistribution(ProbabilityDistribution):
    """ Distribution class for calculating and visualizing a Gaussian
    probability distribution.
    """
    def __init__(self, mu: float = 0.0, var: float = 1.0, sigma: float = 1.0):
        """ The generic distribution has mean of 0 and variance/std of 1,
        indicating a standard normal distribution.
        """
        ProbabilityDistribution.__init__(self, mu, var, sigma)

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
        return 'Mean: {0} \nVariance: {1} \nStandard Deviation: {2}'.format(
            self.mean, self.variance, self.stdev
            )

    def plot_histogram_pdf(self, n_spaces=50):

        """Function to plot the normalized histogram of the data and a plot of the
        probability density function along the same range

        Args:
            n_spaces (int): number of data points

        Returns:
            list: x values for the pdf plot
            list: y values for the pdf plot
        """

        min_range = min(self.data)
        max_range = max(self.data)

        interval = 1.0 * (max_range - min_range) / n_spaces

        points = []
        pdf_values = []

        for i in range(n_spaces):
            point_x = min_range + interval*i
            points.append(point_x)
            pdf_values.append(self.pdf(point_x))

        fig, axes = plt.subplots(2, sharex=True)
        fig.subplots_adjust(hspace=.5)
        axes[0].hist(self.data, density=True)
        axes[0].set_title('Normed Histogram of Data')
        axes[0].set_ylabel('Density')
        axes[1].plot(points, pdf_values)
        axes[1].set_title('Normal Distribution for \n Sample Mean and Sample Standard Deviation')
        axes[0].set_ylabel('Density')
        plt.show()

        return points, pdf_values
