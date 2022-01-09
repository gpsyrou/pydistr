import os
import numpy as np
from pathlib import Path

PROJECT_PATH = Path(os.path.dirname(__file__)).parent
output_loc = PROJECT_PATH / Path('input_files')

num_trials = 10
prob = 0.5
num_samples = 1000

np.random.seed(0)

binomial_values = np.random.binomial(n=num_trials, p=prob, size=num_samples)

with open(os.path.join(output_loc, 'binomial_data.txt'), 'w') as f:
    for item in binomial_values:
        f.write("%s\n" % item)
