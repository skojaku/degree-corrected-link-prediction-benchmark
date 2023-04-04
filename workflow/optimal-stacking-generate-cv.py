from scipy import sparse
import numpy as np
import sys
import pickle as pkl
from OptimalStackingFunctions import *

if "snakemake" in sys.modules:
    input_heldout_feature = snakemake.input["input_heldout_feature"]
    input_train_feature = snakemake.input["input_train_feature"]
    output_cv_dir = snakemake.output["output_cv_dir"]
    # output_cv_x_seen_files = snakemake.input["output_cv_x_seen_files"]
    # output_cv_y_seen_files = snakemake.input["output_cv_y_seen_files"]
    # output_cv_x_unseen_files = snakemake.input["output_cv_x_unseen_files"]
    # output_cv_y_unseen_files = snakemake.input["output_cv_y_unseen_files"]
else:
    input_file = "../data/"
    output_file = "../data/"

with open(input_heldout_feature,"rb") as f:
    df_ho = pkl.load(f)

with open(input_train_feature,"rb") as f:
    df_tr = pkl.load(f)

creat_numpy_files(
    output_cv_dir,
    df_ho,
    df_tr,
)
