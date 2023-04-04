import numpy as np
from os.path import join as j
import itertools
import pandas as pd
from snakemake.utils import Paramspace
import os
from workflow.EmbeddingModels import *
from workflow.NetworkTopologyPredictionModels import *

include: "./workflow/workflow_utils.smk"  # not able to merge this with snakemake_utils.py due to some path breakage issues


# ====================
# Root folder path setting
# ====================

# network file
DATA_DIR = "data"  # set test_data for testing

DERIVED_DIR = j(DATA_DIR, "derived")
NETWORK_DIR = j(DERIVED_DIR, "networks")
RAW_UNPROCESSED_NETWORKS_DIR = j(NETWORK_DIR, "raw")
RAW_PROCESSED_NETWORKS_DIR = j(NETWORK_DIR, "preprocessed")
EMB_DIR = j(DERIVED_DIR, "embedding")
PRED_DIR = j(DERIVED_DIR, "link-prediction")
OPT_STACK_DIR = j(DERIVED_DIR, "optimal_stacking")

# DATA_LIST = [
#     f.split("_")[1].split(".")[0] for f in os.listdir(RAW_UNPROCESSED_NETWORKS_DIR)
# ]
DATA_LIST = [
    "airport-rach"
]
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
params_emb = {"model": list(embedding_models.keys()), "dim": [64]}
#params_emb = {"model": ["node2vec", "deepwalk", "modspec", "leigenmap"], "dim": [64]}
paramspace_emb = to_paramspace(params_emb)


#
# Network-based link prediction
#
params_net_linkpred = {
    "model": list(topology_models.keys())
}
paramspace_net_linkpred = to_paramspace(params_net_linkpred)

# params_cv_files = {
#     "xtrain" : "X_trainE_cv",
#     "fold" : list(range(1,6)),
#     "negativeEdgeSampler": ["uniform", "degreeBiased"],
#     "testEdgeFraction": [0.5],
#     "sampleId": list(range(N_ITERATION)),
# }

# params_cv_files = to_paramspace(params_cv_files)

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

OPT_TRAIN_DATASET_DIR = j(OPT_STACK_DIR, "train_datasets")
OPT_HELDOUT_DATASET_DIR = j(OPT_STACK_DIR, "heldout_datasets")

HELDOUT_NET_FILE_OPTIMAL_STACKING = j(
    OPT_HELDOUT_DATASET_DIR,
    "{data}",
    f"heldout-net_{paramspace_negative_edge_sampler.wildcard_pattern}.npz",
)

TRAIN_NET_FILE_OPTIMAL_STACKING = j(
    OPT_TRAIN_DATASET_DIR,
    "{data}",
    f"train-net_{paramspace_negative_edge_sampler.wildcard_pattern}.npz",
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

TRAIN_FEATURE_MATRIX = j(
    OPT_STACK_DIR,
    "{data}",
    "feature_matrices",
    f"train-feature_{paramspace_negative_edge_sampler.wildcard_pattern}.pkl",
)

HELDOUT_FEATURE_MATRIX = j(
    OPT_STACK_DIR,
    "{data}",
    "feature_matrices",
    f"heldout-feature_{paramspace_negative_edge_sampler.wildcard_pattern}.pkl",
)

CV_DIR = j(OPT_STACK_DIR,
    "{data}",
    "cv",
    f"condition_{paramspace_negative_edge_sampler.wildcard_pattern}",
)

CV_X_SEEN_FILES = j(
    CV_DIR,
    "X_Eseen.npy",
)

CV_Y_SEEN_FILES = j(
    CV_DIR,
    f"y_Eseen.npy",
)

CV_X_UNSEEN_FILES = j(
    CV_DIR,
    f"X_Eunseen.npy",
)

CV_Y_UNSEEN_FILES = j(
    CV_DIR,
    f"y_Eunseen.npy",
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
FIG_AUCROC = j(RESULT_DIR, "figs", "aucroc.pdf")
FIG_DEGSKEW_AUCDIFF = j(RESULT_DIR, "figs", "corr_degskew_aucdiff.pdf")
FIG_NODES_AUCDIFF = j(RESULT_DIR, "figs", "corr_nodes_aucdiff.pdf")


#
#
# RULES
#

rule all:
    input:
        expand(
            LP_ALL_SCORE_FILE,
            data=DATA_LIST,
            **params_emb,
            **params_negative_edge_sampler
        ),
        expand(
            LP_SCORE_EMB_FILE,
            data=DATA_LIST,
            **params_emb,
            **params_negative_edge_sampler
        ),
        expand(
            LP_SCORE_NET_FILE,
            data=DATA_LIST,
            **params_net_linkpred,
            **params_negative_edge_sampler
        ),


rule figs:
    input:
        FIG_AUCROC,
        FIG_DEGSKEW_AUCDIFF,
        FIG_NODES_AUCDIFF,

# ============================
# Cleaning networks
# Gets edge list of GCC as csv
# ============================
rule clean_networks:
    input:
        raw_unprocessed_networks_dir=RAW_UNPROCESSED_NETWORKS_DIR,
        raw_processed_networks_dir=RAW_PROCESSED_NETWORKS_DIR,
    output:
        edge_table_file=EDGE_TABLE_FILE,
    script:
        "workflow/clean_networks.py"

# ============================
# Optimal stacking 
# ============================
rule optimal_stacking_all:
    input:
        expand(
            CV_DIR,
            data=DATA_LIST,
            **params_negative_edge_sampler
        )

rule optimal_stacking_train_heldout_dataset:
    input:
        edge_table_file=EDGE_TABLE_FILE,
    params:
        parameters=paramspace_negative_edge_sampler.instance
    output:
        output_heldout_net_file=HELDOUT_NET_FILE_OPTIMAL_STACKING,
        output_train_net_file=TRAIN_NET_FILE_OPTIMAL_STACKING,
    script:
        "workflow/generate-optimal-stacking-train-heldout-networks.py"

rule optimal_stacking_generate_features:
    input:
        input_original_edge_table_file=EDGE_TABLE_FILE,
        input_heldout_net_file=HELDOUT_NET_FILE_OPTIMAL_STACKING,
        input_train_net_file=TRAIN_NET_FILE_OPTIMAL_STACKING,
    output:
        output_heldout_feature=HELDOUT_FEATURE_MATRIX,
        output_train_feature=TRAIN_FEATURE_MATRIX
    script:
        "workflow/optimal-stacking-topological-features.py"

rule optimal_stacking_generate_cv_files:
    input:
        input_heldout_feature=HELDOUT_FEATURE_MATRIX,
        input_train_feature=TRAIN_FEATURE_MATRIX,
    output:
        output_cv_dir = directory(CV_DIR),
        output_cv_x_seen_files = CV_X_SEEN_FILES,
        output_cv_y_seen_files = CV_Y_SEEN_FILES,
        output_cv_x_unseen_files = CV_X_UNSEEN_FILES,
        output_cv_y_unseen_files = CV_Y_UNSEEN_FILES
    script:
        "workflow/optimal-stacking-generate-cv.py"

# rule optimal_stacking_model_selection:
#     input:
#         input_heldout_feature=HELDOUT_FEATURE_MATRIX,
#     output:
#         output_heldout_feature=HELDOUT_FEATURE_MATRIX,
#         output_train_feature=TRAIN_FEATURE_MATRIX
#     script:
#         "workflow/optimal-stacking-topological-features.py"

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
        output_file=FIG_AUCROC,
    script:
        "workflow/plot-auc-roc.py"


rule plot_aucdiff:
    input:
        auc_results_file=LP_ALL_SCORE_FILE,
        networks_dir=RAW_PROCESSED_NETWORKS_DIR,
    output:
        degskew_outputfile=FIG_DEGSKEW_AUCDIFF,
        nodes_outputfile=FIG_NODES_AUCDIFF,
    script:
        "workflow/plot-NetProp-AucDiff.py"
