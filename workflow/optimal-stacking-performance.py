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
import pickle as pkl

if "snakemake" in sys.modules:
    input_best_rf_params = snakemake.input["input_best_rf_params"]
    input_seen_unseen_data_dir = snakemake.input["input_seen_unseen_data_dir"]
    data_name = snakemake.params["data_name"]
    output_file = snakemake.output["output_file"]
    feature_importance_file = snakemake.output["feature_importance_file"]
    model_file = snakemake.output["model_file"]

else:
    input_file = "../data/"
    output_file = "../data/"

params = pd.read_csv(input_best_rf_params)
n_depth = params["depth"].to_list()[0]
n_est = params["trees"].to_list()[0]

feature_set = [
    "com_ne",
    "ave_deg_net",
    "var_deg_net",
    "ave_clust_net",
    "num_triangles_1",
    "num_triangles_2",
    "page_rank_pers_edges",
    "pag_rank1",
    "pag_rank2",
    "clust_coeff1",
    "clust_coeff2",
    "ave_neigh_deg1",
    "ave_neigh_deg2",
    "eig_cent1",
    "eig_cent2",
    "deg_cent1",
    "deg_cent2",
    "clos_cent1",
    "clos_cent2",
    "betw_cent1",
    "betw_cent2",
    "load_cent1",
    "load_cent2",
    "ktz_cent1",
    "ktz_cent2",
    "pref_attach",
    "LHN",
    "svd_edges",
    "svd_edges_dot",
    "svd_edges_mean",
    "svd_edges_approx",
    "svd_edges_dot_approx",
    "svd_edges_mean_approx",
    "short_path",
    "deg_assort",
    "transit_net",
    "diam_net",
    "jacc_coeff",
    "res_alloc_ind",
    "adam_adar",
    "num_nodes",
    "num_edges",
]

feature_importances, auc_measure, _, _,model = heldout_performance(input_seen_unseen_data_dir, n_depth, n_est)

pd.DataFrame({"score": [auc_measure], "data": data_name}).to_csv(
    output_file, index=False
)

with open(feature_importance_file,"wb") as f:
    feature_importances = dict(zip(feature_set,feature_importances))
    pkl.dump(feature_importances,f)

with open(model_file,"wb") as f:
    pkl.dump(model,f)