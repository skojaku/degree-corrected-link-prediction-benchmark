# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-01-17 04:25:23
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-04-01 07:11:58
from scipy import sparse
import numpy as np
import pandas as pd
import sys
from workflow.models.NetworkModels import *
from RankingModels import *

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    net_file = snakemake.input["net_file"]
    params = snakemake.params["parameters"]
    topK = int(snakemake.params["topK"])
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/"
    output_file = "../data/"

edge_table = pd.read_csv(input_file)
net = sparse.load_npz(net_file)
model = params["model"]

# To ensure that the network is undirected and unweighted.
net = net + net.T
net.data = net.data * 0.0 + 1.0

R = ranking_by_topology(model, net, max_k=topK)
R = R - R.multiply(net)  # remove positive edges from the ranking
R.eliminate_zeros()
R.sort_indices()

# ========================
# Save
# ========================
sparse.save_npz(output_file, R)
