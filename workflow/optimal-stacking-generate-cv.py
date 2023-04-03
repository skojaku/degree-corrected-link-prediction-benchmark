from scipy import sparse
import numpy as np
import sys
import pickle as pkl
from OptimalStackingFunctions import *

if "snakemake" in sys.modules:
    input_heldout_feature = snakemake.output["input_heldout_feature"]
    input_train_file = snakemake.input["input_train_net_file"]
    
    output_cv_dir = snake
    output_cv_x_seen_files = snakemake.input["output_cv_x_seen_files"]
    output_cv_y_seen_files = snakemake.input["output_cv_y_seen_files"]
    output_cv_x_unseen_files = snakemake.input["output_cv_x_unseen_files"]
    output_cv_y_unseen_files = snakemake.input["output_cv_y_unseen_files"]

    output_heldout_feature = snakemake.output["output_heldout_feature"]
    output_train_feature = snakemake.output["output_train_feature"]
else:
    input_file = "../data/"
    output_file = "../data/"


creat_numpy_files(
    output_cv_x_seen_files,
    output_cv_y_seen_files,
    output_cv_x_unseen_files,
    output_cv_y_unseen_files,
    df_ho,
    df_tr,
)
