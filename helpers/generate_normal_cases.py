import numpy as np

mu = 0.0
sigma = 1.0

normal_values = np.random.normal(mu, sigma, size=1000)

with open('normal_data.txt', 'w') as f:
    for item in normal_values:
        f.write("%s\n" % item)
