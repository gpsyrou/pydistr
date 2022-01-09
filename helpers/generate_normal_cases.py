import os
import numpy as np
from pathlib import Path

PROJECT_PATH = Path(os.path.dirname(__file__)).parent
output_loc = PROJECT_PATH / Path('input_files')

mu = 0.0
sigma = 1.0
num_samples = 1000

np.random.seed(0)

normal_values = np.random.normal(mu, sigma, size=num_samples)

with open(os.path.join(output_loc, 'normal_data.txt'), 'w') as f:
    for item in normal_values:
        f.write("%s\n" % item)
