import numpy as np
from os.path import join as j
import itertools
import pandas as pd
from snakemake.utils import Paramspace
import os
from gnn_tools.models import embedding_models

include: "./workflow/workflow_utils.smk"  # not able to merge this with snakemake_utils.py due to some path breakage issues

# Multi partition model
N_SAMPLES = 3
params_mpm = {
    "n": [5000],  # Network size
    "q": [50],  # Number of communities
    "cave": [10, 50],  # average degree
    "mu": ["%.2f" % d for d in np.linspace(0.1, 1, 19)],
    "sample": list(range(N_SAMPLES)),  # Number of samples
}

# Parmaters for the LFR benchmark
params_lfr = { # LFR
    "n": [5000],  # Network size
    "k": [10, 50],  # Average degree
    "tau": [3],  # degree exponent
    "tau2": [1],  # community size exponent
    "minc": [50],  # min community size
    "mu": ["%.2f" % d for d in np.linspace(0.1, 1, 19)],
    "sample": list(range(N_SAMPLES)),  # Number of samples
}

params_clustering = {
    "metric": ["cosine"],
    "clustering": ["kmeans"],
}


# ======================================
# Community Detection Benchmark Datasets
# ======================================

CMD_DATASET_DIR = j(DERIVED_DIR, "community-detection-datasets")

# LFR benchmark
LFR_DIR = j(CMD_DATASET_DIR, "lfr")

LFR_NET_DIR = j(LFR_DIR, "networks")
LFR_EMB_DIR = j(LFR_DIR, "embedding")
LFR_CLUST_DIR = j(LFR_DIR, "clustering")
LFR_EVAL_DIR = j(LFR_DIR, "evaluations")

paramspace_lfr = to_paramspace(params_lfr)
LFR_NET_FILE = j(LFR_NET_DIR, f"net_{paramspace_lfr.wildcard_pattern}.npz")
LFR_NODE_FILE = j(LFR_NET_DIR, f"node_{paramspace_lfr.wildcard_pattern}.npz")

paramspace_lfr_emb = to_paramspace([params_lfr, params_emb])
LFR_EMB_FILE = j(LFR_EMB_DIR, f"{paramspace_lfr_emb.wildcard_pattern}.npz")

paramspace_lfr_com_detect_emb = to_paramspace([params_lfr, params_emb, params_clustering])
LFR_COM_DETECT_EMB_FILE = j(
    LFR_CLUST_DIR, f"clus_{paramspace_lfr_com_detect_emb.wildcard_pattern}.npz"
)

LFR_EVAL_EMB_FILE = j(LFR_EVAL_DIR, f"score_clus_{paramspace_lfr_com_detect_emb.wildcard_pattern}.npz")

# Multi partition model
MPM_DIR = j(CMD_DATASET_DIR, "mpm")

MPM_NET_DIR = j(MPM_DIR, "networks")
MPM_EMB_DIR = j(MPM_DIR, "embedding")
MPM_CLUST_DIR = j(MPM_DIR, "clustering")
MPM_EVAL_DIR = j(MPM_DIR, "evaluations")

paramspace_mpm = to_paramspace(params_mpm)
MPM_NET_FILE = j(MPM_NET_DIR, f"net_{paramspace_mpm.wildcard_pattern}.npz")
MPM_NODE_FILE = j(MPM_NET_DIR, f"node_{paramspace_mpm.wildcard_pattern}.npz")

paramspace_mpm_emb = to_paramspace([params_mpm, params_emb])
MPM_EMB_FILE = j(MPM_EMB_DIR, f"{paramspace_mpm_emb.wildcard_pattern}.npz")

paramspace_mpm_com_detect_emb = to_paramspace([params_mpm, params_emb, params_clustering])
MPM_COM_DETECT_EMB_FILE = j(
    MPM_CLUST_DIR, f"clus_{paramspace_mpm_com_detect_emb.wildcard_pattern}.npz"
)
MPM_EVAL_EMB_FILE = j(MPM_EVAL_DIR, f"score_clus_{paramspace_mpm_com_detect_emb.wildcard_pattern}.npz")

# ======================================
# LFR benchmark
# ======================================

rule all_mpm:
    input:
        expand(MPM_EVAL_EMB_FILE, **params_mpm, **params_emb, **params_clustering)

rule all_lfr:
    input:
        expand(LFR_EVAL_EMB_FILE, **params_lfr, **params_emb, **params_clustering)


rule generate_lfr_net:
    params:
        parameters=paramspace_lfr.instance,
    output:
        output_file=LFR_NET_FILE,
        output_node_file=LFR_NODE_FILE,
    wildcard_constraints:
        data="lfr"
    resources:
        mem="12G",
        time="04:00:00"
    script:
        "workflow/generate-lfr-networks.py"

rule embedding_lfr:
    input:
        net_file=LFR_NET_FILE,
        com_file=LFR_NODE_FILE,
    output:
        output_file=LFR_EMB_FILE,
    params:
        parameters=paramspace_emb.instance,
    script:
        "workflow/embedding.py"

rule kmeans_clustering_lfr:
    input:
        emb_file=LFR_EMB_FILE,
        com_file=LFR_NODE_FILE,
    output:
        output_file=LFR_COM_DETECT_EMB_FILE,
    params:
        parameters=paramspace_lfr_com_detect_emb.instance,
    wildcard_constraints:
        clustering="kmeans",
    resources:
        mem="12G",
        time="01:00:00",
    script:
        "workflow/kmeans-clustering.py"

rule evaluate_communities_lfr:
    input:
        detected_group_file=LFR_COM_DETECT_EMB_FILE,
        com_file=LFR_NODE_FILE,
    output:
        output_file=LFR_EVAL_EMB_FILE,
    resources:
        mem="12G",
        time="00:10:00",
    script:
        "workflow/eval-com-detect-score.py"

# ======================================
# MPM benchmark
# ======================================



rule generate_mpm_net:
    params:
        parameters=paramspace_mpm.instance,
    output:
        output_file=MPM_NET_FILE,
        output_node_file=MPM_NODE_FILE,
    wildcard_constraints:
        data="mpm"
    resources:
        mem="12G",
        time="04:00:00"
    script:
        "workflow/generate-mpm-networks.py"

rule embedding_mpm:
    input:
        net_file=MPM_NET_FILE,
        com_file=MPM_NODE_FILE,
    output:
        output_file=MPM_EMB_FILE,
    params:
        parameters=paramspace_emb.instance,
    script:
        "workflow/embedding.py"

rule kmeans_clustering_mpm:
    input:
        emb_file=MPM_EMB_FILE,
        com_file=MPM_NODE_FILE,
    output:
        output_file=MPM_COM_DETECT_EMB_FILE,
    params:
        parameters=paramspace_mpm_com_detect_emb.instance,
    wildcard_constraints:
        clustering="kmeans",
    resources:
        mem="12G",
        time="01:00:00",
    script:
        "workflow/kmeans-clustering.py"

rule evaluate_communities_mpm:
    input:
        detected_group_file=MPM_COM_DETECT_EMB_FILE,
        com_file=MPM_NODE_FILE,
    output:
        output_file=MPM_EVAL_EMB_FILE,
    resources:
        mem="12G",
        time="00:10:00",
    script:
        "workflow/eval-com-detect-score.py"
