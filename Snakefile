import numpy as np
from os.path import join as j
import itertools
import pandas as pd
from snakemake.utils import Paramspace
import os
#from workflow.models.EmbeddingModels import *
#from workflow.prediction.NetworkTopologyPredictionModels import *

include: "./workflow/workflow_utils.smk"  # not able to merge this with snakemake_utils.py due to some path breakage issues

configfile: "workflow/config.yaml"

# =========================
# Root folder path setting
# =========================

# Main data folder
DATA_DIR = config["data_dir"]  # set test_data for testing
DERIVED_DIR = j(DATA_DIR, "derived")
NETWORK_DIR = j(DERIVED_DIR, "networks")
RAW_UNPROCESSED_NETWORKS_DIR = j(NETWORK_DIR, "raw")
RAW_PROCESSED_NETWORKS_DIR = j(NETWORK_DIR, "preprocessed")
MODEL_DIR = j(DERIVED_DIR, "models")
PRED_DIR = j(DERIVED_DIR, "link-prediction")

# parameter table
PARAM_TABLE_FILE =j(DATA_DIR, "parameter_table.csv")

DATA_LIST = []
if config["small_networks"]:
    with open("workflow/small-networks.json", "r") as f:
        DATA_LIST = json.load(f)
        #DATA_LIST = DATA_LIST[:1]
else:
    DATA_LIST = [
        f.split("_")[1].split(".")[0] for f in os.listdir(RAW_UNPROCESSED_NETWORKS_DIR)
    ]

N_ITERATION = 1
#N_ITERATION = 5

# ============================
# Cleaning networks
# Gets edge list of GCC as csv
# ============================
# Edge table
EDGE_TABLE_FILE = j(NETWORK_DIR, "preprocessed", "{data}", "edge_table.csv")  # train

NET_STAT_FILE = j(
   NETWORK_DIR, "network-stats.csv"
)


rule clean_networks:
    input:
        raw_unprocessed_networks_dir=RAW_UNPROCESSED_NETWORKS_DIR,
        raw_processed_networks_dir=RAW_PROCESSED_NETWORKS_DIR,
    output:
        edge_table_file=EDGE_TABLE_FILE,
    script:
        "workflow/preprocessing/clean_networks.py"

rule calc_network_stats:
    input:
        input_files = expand(EDGE_TABLE_FILE, data = DATA_LIST)
    output:
        output_file = NET_STAT_FILE
    script:
        "workflow/preprocessing/calc-network-stats.py"


# ============================
# Generating benchmark dataset
# ============================
# Train test split
params_train_test_split = {
    "testEdgeFraction": [0.5],
    "sampleId": list(range(N_ITERATION)),
}
paramspace_train_test_split = to_paramspace(paramName = "trainTestSplit", param = params_train_test_split, index=["testEdgeFraction"])

# Negative edge sampler
params_negative_edge_sampler = {
    "negativeEdgeSampler": ["uniform", "degreeBiased"],
}
paramspace_negative_edge_sampler = to_paramspace(paramName = "negativeSampling", param = params_negative_edge_sampler, index ="negativeEdgeSampler")

# Benchmark data files
DATASET_DIR = j(DERIVED_DIR, "datasets")
TRAIN_NET_FILE = j(
    DATASET_DIR,
    "{data}",
    f"train-net_{paramspace_train_test_split.wildcard_pattern}.npz",
)
TARGET_EDGE_TABLE_FILE = j(
    DATASET_DIR,
    "{data}",
    f"targetEdgeTable_{paramspace_train_test_split.wildcard_pattern}_{paramspace_negative_edge_sampler.wildcard_pattern}.csv",
)
TEST_EDGE_TABLE_FILE = j(
    DATASET_DIR,
    "{data}",
    f"testEdgeTable_{paramspace_train_test_split.wildcard_pattern}.csv",
)

# Stat
COMMON_NEIGHBOR_COVERAGE = j(NETWORK_DIR, "common_neighbor_link_coverage.csv")

rule test_train_edge_split:
    input:
        edge_table_file=EDGE_TABLE_FILE,
    params:
        parameters=paramspace_train_test_split.instance,
    output:
        output_train_net_file=TRAIN_NET_FILE,
        output_test_edge_file=TEST_EDGE_TABLE_FILE
    script:
        "workflow/preprocessing/generate-train-test-edge-split.py"

rule generate_target_edges_for_classification:
    input:
        edge_table_file=EDGE_TABLE_FILE,
        train_net_file=TRAIN_NET_FILE,
    params:
        parameters=paramspace_negative_edge_sampler.instance,
    output:
        output_target_edge_table_file=TARGET_EDGE_TABLE_FILE,
    script:
        "workflow/preprocessing/generate-test-edges.py"

rule calc_common_neighbor_edge_coverage:
    input:
        edge_table_files = expand(TEST_EDGE_TABLE_FILE, data = DATA_LIST, **params_train_test_split),
        net_files = expand(TRAIN_NET_FILE, data = DATA_LIST, **params_train_test_split)
    output:
        output_file = COMMON_NEIGHBOR_COVERAGE
    script:
        "workflow/preprocessing/calc-link-coverage-by-distance.py"



# =======================
# Train prediction models
# =======================

# Prediction model
params_model = [
#    {
#        "model": ["line"],
#        "dim":[64],
#        "num_walks":[40],
#        "modelType":["embedding"]
#    },
#    {
#        "model": ["GCN", "GIN", "GAT", "GraphSAGE"],
#        "feature_dim":[64],
#        "dim_h":[64],
#        "num_layers":[2],
#        "in_channels": [64],
#        "hidden_channels": [64],
#        "num_layers": [2],
#        "out_channels": [64],
#        "dim":[64],
#        "modelType":["embedding"]
#    },
    {
        "model": ["seal+GCN", "seal+GIN", "seal+GAT", "seal+GraphSAGE"],
        "feature_dim":[64],
        "dim_h":[64],
        "num_layers":[2],
        "negative_edge_sampler": ["uniform", "degreeBiased"],
        "epochs": 10,
        "hops": 2,
        "batch_size": 50,
        "lr": 1e-3,
        "in_channels": 64,
        "hidden_channels": 64,
        "num_layers": 2,
        "out_channels": 64,
        "modelType":["seal"]
    },
#    {
#        "model":["stacklp"],
#        "modelType":["stacklp"],
#        "negative_edge_sampler": ["uniform", "degreeBiased"],
#        "val_edge_frac":[0.2],
#        "n_train_samples":[10000],
#        "n_cv":[5]
#    },
#    {
#        "model": ["preferentialAttachment", "commonNeighbors", "jaccardIndex", "resourceAllocation", "adamicAdar", "localRandomWalk", "localPathIndex"],
#        "modelType":["network"]
#    }
]
paramspace_model= to_paramspace(paramName = "PredictionModel", param = params_model, index = ["model", "modelType"])

# Model files
MODEL_FILE = j(
    MODEL_DIR,
    "{data}",
    f"model_{paramspace_train_test_split.wildcard_pattern}_{paramspace_model.wildcard_pattern}.pickle",
)

rule train_model:
    input:
        net_file=TRAIN_NET_FILE,
    params:
        parameters=paramspace_model.instance,
    output:
        output_file=MODEL_FILE,
    script:
        "workflow/train-predict/train.py"

# =======================
# Task
# =======================

# classification
RES_LINK_CLASSIFICATION = j(
    PRED_DIR,
    "{data}",
    f"score_task~classification_{paramspace_train_test_split.wildcard_pattern}_{paramspace_negative_edge_sampler.wildcard_pattern}_{paramspace_model.wildcard_pattern}.npz",
)

# Prediction
RES_LINK_PREDICTION = j(
    PRED_DIR,
    "{data}",
    f"score_taks~prediction_{paramspace_train_test_split.wildcard_pattern}_{paramspace_model.wildcard_pattern}.npz",
)

rule link_clasification:
    input:
        train_net_file=TRAIN_NET_FILE,
        model_file = MODEL_FILE,
        test_edge_file = TARGET_EDGE_TABLE_FILE,
    params:
        parameters=paramspace_model.instance,
    output:
        output_file=RES_LINK_CLASSIFICATION,
    script:
        "workflow/train-predict/link-classification.py"


rule link_prediction:
    input:
        train_net_file=TRAIN_NET_FILE,
        model_file = MODEL_FILE,
    params:
        max_all_prediction_network_size=3000,
        parameters=paramspace_model.instance,
    output:
        output_file=RES_LINK_PREDICTION,
    script:
        "workflow/train-predict/link-prediction.py"

# =====================
# Performance evaluation
# =====================
RESULT_DIR = j(DERIVED_DIR, "results")

# Classification
SCORE_CLASSIFICATION_FILE = j(
    RESULT_DIR,
    "auc-roc",
    "{data}",
    f"result_{paramspace_train_test_split.wildcard_pattern}_{paramspace_negative_edge_sampler.wildcard_pattern}_{paramspace_model.wildcard_pattern}.csv",
)

# Ranking
SCORE_PREDICTION_FILE = j(
    RESULT_DIR,
    "ranking",
    "{data}",
    f"result_{paramspace_train_test_split.wildcard_pattern}_{paramspace_model.wildcard_pattern}.csv",
)

# Concatenated
LP_ALL_AUCROC_SCORE_FILE = j(RESULT_DIR, "result_auc_roc.csv")
LP_ALL_RANKING_SCORE_FILE = j(RESULT_DIR, "result_ranking.csv")


rule eval_link_classification:
    input:
        input_file=RES_LINK_CLASSIFICATION,
        test_edge_file = TARGET_EDGE_TABLE_FILE,
    params:
        data_name=lambda wildcards: wildcards.data,
    output:
        output_file=SCORE_CLASSIFICATION_FILE,
    script:
        "workflow/evaluation/eval-link-classification-performance.py"

rule eval_link_prediction:
    input:
        ranking_score_file=RES_LINK_PREDICTION,
        test_edge_table_file = TEST_EDGE_TABLE_FILE
    params:
        data_name=lambda wildcards: wildcards.data,
    output:
        output_file=SCORE_PREDICTION_FILE,
    script:
        "workflow/evaluation/eval-link-prediction-performance.py"

rule concatenate_aucroc_results:
    input:
        input_file_list = expand(
            SCORE_CLASSIFICATION_FILE,
            data=DATA_LIST,
            predictionModel= paramspace_model,
            negativeSampler = paramspace_negative_edge_sampler,
            trainTestSplit = paramspace_train_test_split
        ),
    params:
        param_file = PARAM_TABLE_FILE
    output:
        output_file=LP_ALL_AUCROC_SCORE_FILE,
    script:
        "workflow/concat-results.py"

rule concatenate_ranking_results:
    input:
        input_file_list = expand(
            SCORE_PREDICTION_FILE,
            data=DATA_LIST,
            predictionModel= paramspace_model,
            negativeSampler = paramspace_negative_edge_sampler,
            trainTestSplit = paramspace_train_test_split
        ),
    params:
        param_file = PARAM_TABLE_FILE
    output:
        output_file=LP_ALL_RANKING_SCORE_FILE,
    script:
        "workflow/concat-results.py"

# =======================================================
# Meta analysis of classification/prediction performance
# =======================================================
LP_ALL_QUANTILE_RANKING_FILE = j(RESULT_DIR, "result_quantile_ranking.csv")

rule calc_quantiles:
    input:
        auc_roc_table_file =  LP_ALL_AUCROC_SCORE_FILE,
        ranking_table_file = LP_ALL_RANKING_SCORE_FILE,
        net_stat_file = NET_STAT_FILE
    params:
        param_file= PARAM_TABLE_FILE
    output:
        output_file = LP_ALL_QUANTILE_RANKING_FILE
    script:
        "workflow/evaluation/calc-quantiles.py"

# =====================
# Plot
# =====================
#
##
## Figure 1 and 2
##
#rule calc_deg_deg_plot:
#    input:
#        edge_table_file=EDGE_TABLE_FILE.format(data = "airport-rach"),
#    params:
#        negativeEdgeSampler = lambda wildcards: wildcards.negativeEdgeSampler
#    output:
#        output_file=FIG_DEG_DEG_PLOT,
#    script:
#        "workflow/plots/plot-deg-deg-plot.py"
#
#
#rule calc_deg_skewness_plot:
#    input:
#        auc_roc_table_file =  LP_ALL_AUCROC_SCORE_FILE,
#        ranking_table_file = LP_ALL_RANKING_SCORE_FILE,
#        net_stat_file = NET_STAT_FILE
#    output:
#        output_file=FIG_PERF_VS_KURTOSIS_PLOT,
#    script:
#        "workflow/plots/plot-performance-vs-degree-skewness.py"
#
#rule plot_node2vec_vs_pa_ranking:
#    input:
#        input_file=LP_ALL_QUANTILE_RANKING_FILE,
#    params:
#        negativeEdgeSampler = lambda wildcards: wildcards.negativeEdgeSampler
#    output:
#        output_file=FIG_QUANTILE_RANKING,
#    script:
#        "workflow/plots/plot-ranking-pref-vs-node2vec.py"
#
#rule plot_ranking_correlation:
#    input:
#        input_file=LP_ALL_QUANTILE_RANKING_FILE,
#    params:
#        similarityMetric = lambda wildcards: wildcards.similarityMetric
#    output:
#        output_file=FIG_RANKING_SIMILARITY,
#    script:
#        "workflow/plots/plot-ranking-similarity.py"
#
#rule plot_aucroc:
#    input:
#        input_file=LP_ALL_AUCROC_SCORE_FILE,
#    output:
#        output_file=FIG_AUCROC,
#    script:
#        "workflow/plots/plot-auc-roc.py"
#
#
#rule plot_aucdiff:
#    input:
#        auc_results_file=LP_ALL_AUCROC_SCORE_FILE,
#        networks_dir=RAW_PROCESSED_NETWORKS_DIR,
#    output:
#        degskew_outputfile=FIG_DEGSKEW_AUCDIFF,
#        nodes_outputfile=FIG_NODES_AUCDIFF,
#        degskew_nodesize_outputfile = FIG_DEGSKEW_AUCDIFF_NODESIZE,
#    script:
#        "workflow/plots/plot-NetProp-AucDiff.py"
#
#
#rule plot_prec_recal_f1:
#    input:
#        input_file=LP_ALL_RANKING_SCORE_FILE,
#    output:
#        output_file=FIG_PREC_RECAL_F1,
#    script:
#        "workflow/plots/plot-prec-recall-f1.py"
#

#
#
# RULES
#
save_param_table(PARAM_TABLE_FILE)

rule all:
    input:
        # -----------------------------------
        # Link classification (Check point 1)
        # -----------------------------------
        expand(
            MODEL_FILE,
            data=DATA_LIST,
            predictionModel= paramspace_model,
            negativeSampler = paramspace_negative_edge_sampler,
            trainTestSplit = paramspace_train_test_split
        ),
        # -----------------------------------
        # Link classification (Check point 2)
        # -----------------------------------
        expand(
            RES_LINK_PREDICTION,
            data=DATA_LIST,
            predictionModel= paramspace_model,
            trainTestSplit = paramspace_train_test_split
        ),
        expand(
            RES_LINK_CLASSIFICATION,
            data=DATA_LIST,
            predictionModel= paramspace_model,
            negativeSampler = paramspace_negative_edge_sampler,
            trainTestSplit = paramspace_train_test_split
        ),
        # -----------------------------------
        # Evaluation (Check point 3)
        # -----------------------------------
        expand(
            SCORE_PREDICTION_FILE,
            data=DATA_LIST,
            predictionModel= paramspace_model,
            negativeSampler = paramspace_negative_edge_sampler,
            trainTestSplit = paramspace_train_test_split
        ),
        expand(
            SCORE_CLASSIFICATION_FILE,
            data=DATA_LIST,
            predictionModel= paramspace_model,
            negativeSampler = paramspace_negative_edge_sampler,
            trainTestSplit = paramspace_train_test_split
        ),
        #
        # -----------------------------------
        # Plot (Check point 4)
        # -----------------------------------
        LP_ALL_QUANTILE_RANKING_FILE

        #
        # Network stats (Check point 2)
        #
        #NET_STAT_FILE,
        #COMMON_NEIGHBOR_COVERAGE,
        #
        # Link ranking (Check point 3)
        #
#        expand(
#            RANK_SCORE_MODEL_FILE,
#            data=DATA_LIST,
#            **params_model,
#            **params_train_test_split
#        ),
#        expand(
#            RANK_SCORE_NET_FILE,
#            data=DATA_LIST,
#            **params_net_linkpred,
#            **params_train_test_split
#        )
        #
        # Quantile ranking file (Checkpoint 4)
        #
        #LP_ALL_QUANTILE_RANKING_FILE



# ====================
# Output
# ====================
FIG_AUCROC = j(RESULT_DIR, "figs", "aucroc.pdf")
FIG_DEGSKEW_AUCDIFF = j(RESULT_DIR, "figs", "corr_degskew_aucdiff.pdf")
FIG_NODES_AUCDIFF = j(RESULT_DIR, "figs", "corr_nodes_aucdiff.pdf")
FIG_DEGSKEW_AUCDIFF_NODESIZE = j(RESULT_DIR, "figs", "corr_degskew_aucdiff_nodesize.pdf")
FIG_PREC_RECAL_F1 =j(RESULT_DIR, "figs", "prec-recall-f1.pdf")
FIG_DEG_DEG_PLOT =j(RESULT_DIR, "figs", "deg_deg_plot_negativeEdgeSampler~{negativeEdgeSampler}.pdf")
FIG_QUANTILE_RANKING=j(RESULT_DIR, "figs", "quantile_ranking_negativeEdgeSampler~{negativeEdgeSampler}.pdf")
FIG_PERF_VS_KURTOSIS_PLOT=j(RESULT_DIR, "figs", "performance_vs_degree_kurtosis.pdf")
FIG_RANKING_SIMILARITY=j(RESULT_DIR, "figs", "ranking-similarity-similarityMetric~{similarityMetric}.pdf")

rule figs:
    input:
        expand(FIG_DEG_DEG_PLOT, **params_negative_edge_sampler),
        expand(FIG_QUANTILE_RANKING, **params_negative_edge_sampler),
        expand(FIG_RANKING_SIMILARITY, similarityMetric = ["RBO", "Spearmanr"]),
        FIG_PERF_VS_KURTOSIS_PLOT,
        FIG_AUCROC,
        #FIG_DEGSKEW_AUCDIFF,
        #FIG_NODES_AUCDIFF,
        #FIG_DEGSKEW_AUCDIFF_NODESIZE,
        #FIG_PREC_RECAL_F1

rule _all:
    input:
        #COMMON_NEIGHBOR_COVERAGE,
        LP_ALL_QUANTILE_RANKING_FILE