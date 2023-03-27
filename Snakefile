import numpy as np
from os.path import join as j
import itertools
import pandas as pd
from snakemake.utils import Paramspace
include: "./workflow_utils.smk" # not able to merge this with snakemake_utils.py due to some path breakage issues

# ====================
# Root folder path setting
# ====================

# network file
DERIVED_DIR = j("data", "derived")
NETWORK_DIR = j(DERIVED_DIR, "networks")

DATA_LIST = ["airport", "polbook", "polblog"]
N_ITERATION = 5

# ====================
# Configuration
# ====================
params_negative_edge_sampler = {
    "edgeSampling":["uniform", "degreeBiased"],
    "testEdgeFrac":[0.1, 0.5],
    "sampleId":list(range(N_ITERATION)),
}
paramspace_negative_edge_sampler = to_paramspace(params_negative_edge_sampler)

params_emb = {
    "model":["node2vec", "deepwalk", "modularity", "leigenmap"],
}
paramspace_emb = to_paramspace(params_emb)
# =============================
# Networks & Benchmark Datasets
# =============================

# Edge table
EDGE_TABLE_FILE = j(NETWORK_DIR, "raw", "{data}", "edge_table.csv") # train

# Benchmark
DATASET_DIR = j(DERIVED_DIR, "datasets")
LP_DATASET_FILE = j(DATASET_DIR, "{data}", f"{paramspace.wildcard_pattern}.csv")

# ====================
# Evaluation
# ====================
RESULT_DIR = j(DERIVED_DIR, "results")

# AUC-ROC
LP_SCORE_FILE = j(RESULT_DIR, "auc_roc", "{data}", f"result_{paramspace.wildcard_pattern}_{paramspace_emb.wildcard_pattern}.csv")
LP_ALL_SCORE_FILE = j(RESULT_DIR, "result_auc_roc.csv")

# ====================
# Output
# ====================

#
#
# RULES
#
#
rule all:
    input:
        expand(LP_ALL_SCORE_FILE, data = DATA_LIST, **params_emb, **params_negative_edge_sampler),


# ============================
# Generating benchmark dataset
# ============================
rule generate_link_prediction_dataset:
    input:
        edge_table_file = EDGE_TABLE_FILE,
    params:
        parameters=paramspace_negative_edge_sampler.instance,
    output:
        output_file = LP_DATASET_FILE
    script:
        "workflow/generate-link-prediction-dataset.py"


# =====================
# Evaluation
# =====================
rule eval_link_prediction:
    input:
        input_file = LP_DATASET_FILE,
    params:
        emb_file = lambda wildcards: "{root}/{data}/{data}_{sampleId}/{data}".format(root=SRC_DATA_ROOT, data=wildcards.data, sampleId=wildcards.sampleId)+MODEL2EMBFILE_POSTFIX[wildcards.model] # not ideal but since the file names are different, I generate the file name in the script and load the corresponding file.
    output:
        output_file = LP_SCORE_FILE
    script:
        "workflow/evaluate-lp-performance.py"

rule concatenate_results:
    input:
        input_file_list = expand(LP_SCORE_FILE, **lp_benchmark_params, data = DATA_LIST),
    output:
        output_file = LP_ALL_SCORE_FILE
    script:
        "workflow/concat-results.py"