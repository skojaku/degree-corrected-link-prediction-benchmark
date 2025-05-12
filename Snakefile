import numpy as np
from os.path import join as j
import itertools
import pandas as pd
from snakemake.utils import Paramspace
import os
from gnn_tools.models import embedding_models
from workflow.NetworkTopologyPredictionModels import *

include: "./workflow/workflow_utils.smk"  # not able to merge this with snakemake_utils.py due to some path breakage issues

configfile: "workflow/config.yaml"

# ====================
# Root folder path setting
# ====================

# network file
DATA_DIR = config["data_dir"]  # set test_data for testing

DERIVED_DIR = j(DATA_DIR, "derived")
RAW_UNPROCESSED_NETWORKS_DIR = j(DATA_DIR, "raw")
STAT_DIR = j(DATA_DIR, "stats")
RAW_PROCESSED_NETWORKS_DIR = j(DATA_DIR, "preprocessed")
EMB_DIR = j(DERIVED_DIR, "embedding")
PRED_DIR = j(DERIVED_DIR, "link-prediction")
OPT_STACK_DIR = j(DERIVED_DIR, "optimal_stacking")
FIG_DIR =j("figs")

#All networks
DATA_LIST = [
    f.split("net_")[1].split(".")[0] for f in os.listdir(RAW_UNPROCESSED_NETWORKS_DIR)
]
DATA_LIST = [d for d in DATA_LIST if "ogbl" not in d]

# Small networks
# Comment out if you want to run for all networks
if config["small_networks"]:
    with open("workflow/small-networks.json", "r") as f:
        DATA_LIST = json.load(f)

N_ITERATION =  5

# ====================
# Configuration
# ====================

#
# Negative edge sampler
#
params_train_test_split = {
    "testEdgeFraction": [0.25],
    "sampleId": list(range(N_ITERATION)),
}
paramspace_train_test_split = to_paramspace(params_train_test_split)


params_negative_edge_sampler = {
    "negativeEdgeSampler": ["heart"],
    #"negativeEdgeSampler": ["uniform", "degreeBiased", "heart"],
}
paramspace_negative_edge_sampler = to_paramspace(params_negative_edge_sampler)


#
# Network embedding
#
MODEL_LIST = list(embedding_models.keys())
#MODEL_LIST = ["GraphSAGE", "GIN", "GAT", "GCN", "dcGraphSAGE", "dcGIN", "dcGAT", "dcGCN"] # added to limit
MODEL_LIST = [m for m in MODEL_LIST if m not in ["EdgeCNN", "dcEdgeCNN"]]
params_emb = {"model": MODEL_LIST, "dim": [128]}
paramspace_emb = to_paramspace(params_emb)


#
# Network-based link prediction
#
params_net_linkpred = {
    "model": list(topology_models.keys())
}
paramspace_net_linkpred = to_paramspace(params_net_linkpred)



# =============================
# Networks
# =============================

# Edge table
EDGE_TABLE_FILE = j(RAW_PROCESSED_NETWORKS_DIR, "{data}", "edge_table.csv")  # train


# =============================
# Link Prediction Benchmark Datasets
# =============================

# Link prediction benchmark
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

#
# Network statistics
#
NET_STAT_FILE = j(
   STAT_DIR, "network-stats.csv"
)

#
# Embedding
#
EMB_FILE = j(
    EMB_DIR,
    "{data}",
    f"emb_{paramspace_train_test_split.wildcard_pattern}_{paramspace_emb.wildcard_pattern}.npz",
)
# classification
PRED_SCORE_EMB_FILE = j(
    PRED_DIR,
    "{data}",
    f"score_basedOn~emb_{paramspace_train_test_split.wildcard_pattern}_{paramspace_negative_edge_sampler.wildcard_pattern}_{paramspace_emb.wildcard_pattern}.csv",
)

#
# Topology-based
#
PRED_SCORE_NET_FILE = j(
    PRED_DIR,
    "{data}",
    f"score_basedOn~net_{paramspace_train_test_split.wildcard_pattern}_{paramspace_negative_edge_sampler.wildcard_pattern}_{paramspace_net_linkpred.wildcard_pattern}.csv",
)

# ====================
# Evaluation
# ====================
RESULT_DIR = j(DERIVED_DIR, "results")

# Classification
LP_SCORE_EMB_FILE = j(
    RESULT_DIR,
    "auc-roc",
    "{data}",
    f"result_basedOn~emb_{paramspace_train_test_split.wildcard_pattern}_{paramspace_negative_edge_sampler.wildcard_pattern}_{paramspace_emb.wildcard_pattern}.csv",
)
LP_SCORE_NET_FILE = j(
    RESULT_DIR,
    "auc-roc",
    "{data}",
    f"result_basedOn~net_{paramspace_train_test_split.wildcard_pattern}_{paramspace_negative_edge_sampler.wildcard_pattern}_{paramspace_net_linkpred.wildcard_pattern}.csv",
)

# Retrieval task
RT_SCORE_EMB_FILE = j(
    RESULT_DIR,
    "retrieval",
    "{data}",
    f"result_basedOn~emb_{paramspace_train_test_split.wildcard_pattern}_{paramspace_emb.wildcard_pattern}.csv",
)
RT_SCORE_NET_FILE = j(
    RESULT_DIR,
    "retrieval",
    "{data}",
    f"result_basedOn~net_{paramspace_train_test_split.wildcard_pattern}_{paramspace_net_linkpred.wildcard_pattern}.csv",
)

RT_ALL_SCORE_FILE = j(RESULT_DIR, "result_retrieval.csv")

# Concatenated results
LP_ALL_AUCROC_SCORE_FILE = j(RESULT_DIR, "result_auc_roc.csv")

LP_SCORE_OPT_STACK_FILE = j(
    OPT_STACK_DIR,
    "auc-roc",
    "{data}",
    f"result_basedOn~optstack_{paramspace_train_test_split.wildcard_pattern}_{paramspace_negative_edge_sampler.wildcard_pattern}.csv"
)

BEST_RF_FEATURES = j(
    OPT_STACK_DIR,
    "feature-importance",
    "{data}",
    f"bestfeatures-rf_basedOn~optstack_{paramspace_train_test_split.wildcard_pattern}_{paramspace_negative_edge_sampler.wildcard_pattern}.pkl",
)

LP_ALL_SCORE_OPT_STACK_FILE = j(RESULT_DIR, "result_opt_stack_auc_roc.csv")

# ====================
# Output
# ====================
FIG_AUCROC = j(FIG_DIR, "aucroc.pdf")
FIG_AUCROC_UNIFORM = j(FIG_DIR, "aucroc_uniform.pdf")
FIG_DEG_DEG_PLOT =j(FIG_DIR, "deg_deg_plot_negativeEdgeSampler~{negativeEdgeSampler}.pdf")

params_rbo = {
    "rbop": ["0.1", "0.25", "0.5", "0.75", "0.9", "0.98"],
    "topk": ["10", "50", "100"],
    "focal_score": ["vp", "prec", "rec"],
}
paramspace_rbo = to_paramspace(params_rbo)
FIG_RBO = j(FIG_DIR, "rbo", f"rbo-{paramspace_rbo.wildcard_pattern}.pdf")
FIG_RANK_CHANGE = j(FIG_DIR, "rank-change.pdf")
FIG_DEG_DIST_POS_NEG_EDGES =j(FIG_DIR, "deg-pos-neg-edges.pdf")
FIG_AUCROC_LOGNORMAL = j(FIG_DIR, "auc-roc-log-normal.pdf")
#
#
# RULES
#
rule all:
    input:
        #
        # All results
        #
        LP_ALL_AUCROC_SCORE_FILE,
	    RT_ALL_SCORE_FILE,
        #
        # Generate the link prediction benchmark (Check point 1)
        # [Implement from here] @ vision
        expand(TARGET_EDGE_TABLE_FILE, data=DATA_LIST, **params_train_test_split, **params_negative_edge_sampler),
        #
        # Network stats (Check point 2)
        #
        NET_STAT_FILE,
        #AUCROC_PA_POWERLAW_FILE,
        #
        # Link classification (Check point 3)
        #
        expand(
            LP_SCORE_EMB_FILE,
            data=DATA_LIST,
            **params_emb,
            **params_negative_edge_sampler,
            **params_train_test_split
        ),
        expand(
            LP_SCORE_NET_FILE,
            data=DATA_LIST,
            **params_net_linkpred,
            **params_negative_edge_sampler,
            **params_train_test_split
        ),


rule figs:
    input:
        expand(FIG_DEG_DEG_PLOT, **params_negative_edge_sampler),
        FIG_AUCROC,
        FIG_RANK_CHANGE,
        expand(FIG_RBO, **params_rbo),
        #FIG_PERF_VS_KURTOSIS_PLOT,
        #FIG_DEG_DIST_POS_NEG_EDGES,
	    #FIG_AUCROC_LOGNORMAL


# ============================
# Cleaning networks
# Gets edge list of GCC as csv
# ============================
rule clean_networks:
    input:
        raw_unprocessed_networks_dir=RAW_UNPROCESSED_NETWORKS_DIR,
        #raw_processed_networks_dir=RAW_PROCESSED_NETWORKS_DIR,
    output:
        edge_table_file=EDGE_TABLE_FILE,
    script:
        "workflow/clean_networks.py"

rule calc_network_stats:
    input:
        input_files = expand(EDGE_TABLE_FILE, data = DATA_LIST)
    output:
        output_file = NET_STAT_FILE
    script:
        "workflow/calc-network-stats.py"

#rule calc_aucroc_pa_powerlaw:
#    output:
#        output_file = AUCROC_PA_POWERLAW_FILE
#    script:
#        "workflow/calc-aucroc-pa-powerlaw.py"

# ============================
# Generating benchmark dataset
# ============================
rule generate_link_prediction_dataset:
    input:
        edge_table_file=EDGE_TABLE_FILE,
    params:
        parameters=paramspace_train_test_split.instance,
    output:
        output_train_net_file=TRAIN_NET_FILE,
        output_test_edge_file=TEST_EDGE_TABLE_FILE
    script:
        "workflow/generate-train-test-edge-split.py"

rule train_test_edge_split:
    input:
        edge_table_file=EDGE_TABLE_FILE,
        train_net_file=TRAIN_NET_FILE,
    params:
        parameters=paramspace_negative_edge_sampler.instance,
    output:
        output_target_edge_table_file=TARGET_EDGE_TABLE_FILE,
    script:
        "workflow/generate-test-edges.py"


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


#
# Positive vs Negative
#
rule embedding_link_prediction:
    input:
        input_file=TARGET_EDGE_TABLE_FILE,
        emb_file=EMB_FILE,
        net_file = TRAIN_NET_FILE,
    params:
        parameters=paramspace_emb.instance,
    output:
        output_file=PRED_SCORE_EMB_FILE,
    script:
        "workflow/embedding-link-prediction.py"

# ==============================
# Prediction based on networks
# ==============================

#
# Positive vs Negative
#
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


#
# Positive vs Negative edges
#
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

rule concatenate_aucroc_results:
    input:
        input_file_list=expand(
            LP_SCORE_EMB_FILE,
            data=DATA_LIST,
            **params_emb,
            **params_negative_edge_sampler,
            **params_train_test_split
        )
        + expand(
            LP_SCORE_NET_FILE,
            data=DATA_LIST,
            **params_net_linkpred,
            **params_negative_edge_sampler,
            **params_train_test_split
        )
    output:
        output_file=LP_ALL_AUCROC_SCORE_FILE,
    script:
        "workflow/concat-results.py"

#
# Retrieval task
#
rule eval_link_retrieval_embedding:
    input:
        train_net_file = TRAIN_NET_FILE,
        test_edge_file = TEST_EDGE_TABLE_FILE,
        emb_file = EMB_FILE,
    params:
        data_name=lambda wildcards: wildcards.data,
        model=lambda wildcards: wildcards.model,
        model_type="embedding",
    output:
        output_file=RT_SCORE_EMB_FILE,
    script:
        "workflow/run-link-retrieval-benchmark.py"

rule eval_link_retrieval_networks:
    input:
        train_net_file = TRAIN_NET_FILE,
        test_edge_file = TEST_EDGE_TABLE_FILE,
    params:
        data_name=lambda wildcards: wildcards.data,
        model=lambda wildcards: wildcards.model,
        model_type="topology",
    output:
        output_file=RT_SCORE_NET_FILE,
    script:
        "workflow/run-link-retrieval-benchmark.py"

rule all_retrieval:
    input:
        RT_ALL_SCORE_FILE,

rule concatenate_results_retrieval:
    input:
        input_file_list=expand(RT_SCORE_EMB_FILE, data=DATA_LIST, **params_emb, **params_train_test_split) + expand(RT_SCORE_NET_FILE, data=DATA_LIST, **params_net_linkpred, **params_train_test_split),
    output:
        output_file=RT_ALL_SCORE_FILE,
    script:
        "workflow/concat-results.py"

# =====================
# Plot
# =====================

rule plot_aucroc:
    input:
        auc_roc_table_file=LP_ALL_AUCROC_SCORE_FILE,
        netstat_table_file=NET_STAT_FILE,
    output:
        output_file=FIG_AUCROC,
        output_file_uniform=FIG_AUCROC_UNIFORM,
    script:
        "workflow/plot-auc-roc.py"
