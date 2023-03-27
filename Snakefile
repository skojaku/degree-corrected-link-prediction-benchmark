import numpy as np
from os.path import join as j
import itertools
import pandas as pd
from snakemake.utils import Paramspace
include: "./workflow/workflow_utils.smk" # not able to merge this with snakemake_utils.py due to some path breakage issues

# ====================
# Root folder path setting
# ====================

# network file
DERIVED_DIR = j("data", "derived")
NETWORK_DIR = j(DERIVED_DIR, "networks")
EMB_DIR = j(DERIVED_DIR, "embedding")

DATA_LIST = ["airport", "polblog"]
N_ITERATION = 5

# ====================
# Configuration
# ====================
params_negative_edge_sampler = {
    "negativeEdgeSampler":["uniform", "degreeBiased"],
    "testEdgeFraction":[0.1, 0.5],
    "sampleId":list(range(N_ITERATION)),
}
paramspace_negative_edge_sampler = to_paramspace(params_negative_edge_sampler)

params_emb = {
    "model":["node2vec", "deepwalk", "modspec", "leigenmap", "degree"],
    "dim":[64]
}
paramspace_emb = to_paramspace(params_emb)

# =============================
# Networks & Benchmark Datasets
# =============================

# Edge table
EDGE_TABLE_FILE = j(NETWORK_DIR, "raw", "{data}", "edge_table.csv") # train

# Benchmark
DATASET_DIR = j(DERIVED_DIR, "datasets")
TRAIN_NET_FILE = j(DATASET_DIR, "{data}", f"train-net_{paramspace_negative_edge_sampler.wildcard_pattern}.npz")
TARGET_EDGE_TABLE_FILE = j(DATASET_DIR, "{data}", f"targetEdgeTable_{paramspace_negative_edge_sampler.wildcard_pattern}.csv")


# ====================
# Intermediate files
# ====================
EMB_FILE = j(EMB_DIR, "{data}", f"emb_{paramspace_negative_edge_sampler.wildcard_pattern}_{paramspace_emb.wildcard_pattern}.npz")

# ====================
# Evaluation
# ====================
RESULT_DIR = j(DERIVED_DIR, "results")

# AUC-ROC
LP_SCORE_FILE = j(RESULT_DIR, "auc-roc", "{data}", f"result_{paramspace_negative_edge_sampler.wildcard_pattern}_{paramspace_emb.wildcard_pattern}.csv")
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
        output_train_net_file = TRAIN_NET_FILE,
        output_target_edge_table_file = TARGET_EDGE_TABLE_FILE
    script:
        "workflow/generate-link-prediction-dataset.py"

# =====================
# Embedding
# =====================
rule embedding:
    input:
        net_file = TRAIN_NET_FILE
    params:
        parameters=paramspace_emb.instance,
    output:
        output_file = EMB_FILE
    script:
        "workflow/embedding.py"

# =====================
# Evaluation
# =====================
rule eval_link_prediction:
    input:
        input_file = TARGET_EDGE_TABLE_FILE,
        emb_file = EMB_FILE
    output:
        output_file = LP_SCORE_FILE
    script:
        "workflow/evaluate-lp-performance.py"

rule concatenate_results:
    input:
        input_file_list = expand(LP_SCORE_FILE, data = DATA_LIST, **params_emb, **params_negative_edge_sampler),
    output:
        output_file = LP_ALL_SCORE_FILE
    script:
        "workflow/concat-results.py"