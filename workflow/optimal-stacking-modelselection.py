from scipy import sparse
import numpy as np
import sys
import pandas as pd
from OptimalStackingFunctions import *

if "snakemake" in sys.modules:
    input_cv_dir = snakemake.input["input_cv_dir"]
    output_best_rf_params = snakemake.output["output_best_rf_params"]
else:
    input_file = "../data/"
    output_file = "../data/"

n_depths = [3, 6]  # here is a sample search space
n_ests = [25, 50, 100]  # here is a sample search space

n_depth, n_est = model_selection(input_cv_dir, input_cv_dir, n_depths, n_ests)

df = pd.DataFrame({"cv": [input_cv_dir], "depth": [n_depth], "trees": [n_est]})

df.to_csv(output_best_rf_params)