# /*
#  * @Author: Rachith Aiyappa
#  * @Date: 2023-04-04 12:36:01
#  * @Last Modified by:   Rachith
#  * @Last Modified time: 2023-04-04 12:36:01
#  */

"""
Get auc scores by evaluating on heldout data
"""

from scipy import sparse
import numpy as np
import sys
import pandas as pd
from OptimalStackingFunctions import *

if "snakemake" in sys.modules:
    input_best_rf_params = snakemake.input["input_best_rf_params"]
    input_seen_unseen_data_dir = snakemake.input["input_seen_unseen_data_dir"]
    data_name = snakemake.params["data_name"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/"
    output_file = "../data/"

params = pd.read_csv(input_best_rf_params)
n_depth = params["depth"].to_list()[0]
n_est = params["trees"].to_list()[0]

auc_measure, _, _ = heldout_performance(input_seen_unseen_data_dir, n_depth, n_est)

pd.DataFrame({"score": [auc_measure], "data": data_name}).to_csv(
    output_file, index=False
)
