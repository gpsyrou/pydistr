
class ProbabilityDistribution:
    """ Generic class for computing and visualizing a probability distribution.
    The class provides support for both Discrete and Continuous probability
    distributions.

    Attributes
    ----------
        mean: Mean value of the distribution
        stdev: Standard deviation of the distribution
        data_list: Input values for the distribution, as read from a text file
        """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        self.mean = mu
        self.stdev = sigma
        self.data = []

    def read_data_from_text_file(self, filename: str):
        """ Method to read the data points from a text file ("filename"), and
        save the results in the data attribute.

        The text file should have one data point (float type) per line.

        Arguments
        ---------
            filename: Name(path) to the file that contains the data points.
        """
        with open(file=filename, mode='r') as file:
            for line in file:
                self.data.append(float(line))
        file.close()
