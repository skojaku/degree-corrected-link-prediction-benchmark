import numpy as np
from os.path import join as j
import itertools
import pandas as pd
from snakemake.utils import Paramspace
import os


include: "./workflow/workflow_utils.smk"  # not able to merge this with snakemake_utils.py due to some path breakage issues


# ====================
# Root folder path setting
# ====================

# network file
DATA_DIR = "data" # set test_data for testing

DERIVED_DIR = j(DATA_DIR, "derived")
NETWORK_DIR = j(DERIVED_DIR, "networks")
RAW_UNPROCESSED_NETWORKS_DIR = j(NETWORK_DIR,"raw")
RAW_PROCESSED_NETWORKS_DIR = j(NETWORK_DIR,"preprocessed")
EMB_DIR = j(DERIVED_DIR, "embedding")
PRED_DIR = j(DERIVED_DIR, "link-prediction")

DATA_LIST = [f.split("_")[1].split('.')[0] for f in os.listdir(RAW_UNPROCESSED_NETWORKS_DIR)]
N_ITERATION = 5

# ====================
# Configuration
# ====================

#
# Negative edge sampler
#
params_negative_edge_sampler = {
    "negativeEdgeSampler": ["uniform", "degreeBiased"],
    "testEdgeFraction": [0.5],
    "sampleId": list(range(N_ITERATION)),
}
paramspace_negative_edge_sampler = to_paramspace(params_negative_edge_sampler)

#
# Network embedding
#
params_emb = {"model": ["node2vec", "deepwalk", "modspec", "leigenmap"], "dim": [64]}
paramspace_emb = to_paramspace(params_emb)


#
# Network-based link prediction
#
params_net_linkpred = {
    "model": [
        "preferentialAttachment",
        "commonNeighbors",
        "jaccardIndex",
        "resourceAllocation",
        "adamicAdar",
    ],
}
paramspace_net_linkpred = to_paramspace(params_net_linkpred)

# =============================
# Networks & Benchmark Datasets
# =============================

# Edge table
EDGE_TABLE_FILE = j(NETWORK_DIR, "preprocessed", "{data}", "edge_table.csv")  # train

# Benchmark
DATASET_DIR = j(DERIVED_DIR, "datasets")
TRAIN_NET_FILE = j(
    DATASET_DIR,
    "{data}",
    f"train-net_{paramspace_negative_edge_sampler.wildcard_pattern}.npz",
)
TARGET_EDGE_TABLE_FILE = j(
    DATASET_DIR,
    "{data}",
    f"targetEdgeTable_{paramspace_negative_edge_sampler.wildcard_pattern}.csv",
)


# ====================
# Intermediate files
# ====================

#
# Embedding
#
EMB_FILE = j(
    EMB_DIR,
    "{data}",
    f"emb_{paramspace_negative_edge_sampler.wildcard_pattern}_{paramspace_emb.wildcard_pattern}.npz",
)
PRED_SCORE_EMB_FILE = j(
    PRED_DIR,
    "{data}",
    f"score_basedOn~emb_{paramspace_negative_edge_sampler.wildcard_pattern}_{paramspace_emb.wildcard_pattern}.csv",
)
PRED_SCORE_NET_FILE = j(
    PRED_DIR,
    "{data}",
    f"score_basedOn~net_{paramspace_negative_edge_sampler.wildcard_pattern}_{paramspace_net_linkpred.wildcard_pattern}.csv",
)

# ====================
# Evaluation
# ====================
RESULT_DIR = j(DERIVED_DIR, "results")

# AUC-ROC
LP_SCORE_EMB_FILE = j(
    RESULT_DIR,
    "auc-roc",
    "{data}",
    f"result_basedOn~emb_{paramspace_negative_edge_sampler.wildcard_pattern}_{paramspace_emb.wildcard_pattern}.csv",
)
LP_SCORE_NET_FILE = j(
    RESULT_DIR,
    "auc-roc",
    "{data}",
    f"result_basedOn~net_{paramspace_negative_edge_sampler.wildcard_pattern}_{paramspace_net_linkpred.wildcard_pattern}.csv",
)
LP_ALL_SCORE_FILE = j(RESULT_DIR, "result_auc_roc.csv")

# ====================
# Output
# ====================
FIG_AUCROC = j("figs", "aucroc.pdf")

#
#
# RULES
#
#
rule all:
    input:
        expand(
            LP_ALL_SCORE_FILE,
            data=DATA_LIST,
            **params_emb,
            **params_negative_edge_sampler
        ),

rule figs:
    input:
        FIG_AUCROC

# ============================
# Cleaning networks
# Gets edge list of GCC as csv
# ============================
rule clean_networks:
    input:
        raw_unprocessed_networks_dir = RAW_UNPROCESSED_NETWORKS_DIR,
        raw_processed_networks_dir = RAW_PROCESSED_NETWORKS_DIR,
    output:
        edge_table_file = EDGE_TABLE_FILE,
    script:
        "workflow/clean_networks.py"


# ============================
# Generating benchmark dataset
# ============================
rule generate_link_prediction_dataset:
    input:
        edge_table_file=EDGE_TABLE_FILE,
    params:
        parameters=paramspace_negative_edge_sampler.instance,
    output:
        output_train_net_file=TRAIN_NET_FILE,
        output_target_edge_table_file=TARGET_EDGE_TABLE_FILE,
    script:
        "workflow/generate-link-prediction-dataset.py"


# ==============================
# Prediction based on embedding
# ==============================
rule embedding:
    input:
        net_file=TRAIN_NET_FILE,
    params:
        parameters=paramspace_emb.instance,
    output:
        output_file=EMB_FILE,
    script:
        "workflow/embedding.py"


rule embedding_link_prediction:
    input:
        input_file=TARGET_EDGE_TABLE_FILE,
        emb_file=EMB_FILE,
    params:
        parameters=paramspace_emb.instance,
    output:
        output_file=PRED_SCORE_EMB_FILE,
    script:
        "workflow/embedding-link-prediction.py"


# ==============================
# Prediction based on networks
# ==============================
rule network_link_prediction:
    input:
        input_file=TARGET_EDGE_TABLE_FILE,
        net_file=TRAIN_NET_FILE,
    params:
        parameters=paramspace_net_linkpred.instance,
    output:
        output_file=PRED_SCORE_NET_FILE,
    script:
        "workflow/network-link-prediction.py"


# =====================
# Evaluation
# =====================
rule eval_link_prediction_embedding:
    input:
        input_file=PRED_SCORE_EMB_FILE,
    params:
        data_name=lambda wildcards: wildcards.data,
    output:
        output_file=LP_SCORE_EMB_FILE,
    script:
        "workflow/eval-link-prediction-performance.py"


rule eval_link_prediction_networks:
    input:
        input_file=PRED_SCORE_NET_FILE,
    params:
        data_name=lambda wildcards: wildcards.data,
    output:
        output_file=LP_SCORE_NET_FILE,
    script:
        "workflow/eval-link-prediction-performance.py"


rule concatenate_results:
    input:
        input_file_list=expand(
            LP_SCORE_EMB_FILE,
            data=DATA_LIST,
            **params_emb,
            **params_negative_edge_sampler
        )
        + expand(
            LP_SCORE_NET_FILE,
            data=DATA_LIST,
            **params_net_linkpred,
            **params_negative_edge_sampler
        ),
    output:
        output_file=LP_ALL_SCORE_FILE,
    script:
        "workflow/concat-results.py"

# =====================
# Plot
# =====================
rule plot_aucroc:
    input:
        input_file=LP_ALL_SCORE_FILE,
    output:
        output_file=FIG_AUCROC
    script:
        "workflow/plot-auc-roc.py"