# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-01-17 03:57:05
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-03-27 18:21:14
import numpy as np
import pandas as pd
from scipy import sparse
import sys
from tqdm import tqdm
from linkpred.LinkPredictionDataset import LinkPredictionDataset

if "snakemake" in sys.modules:
    edge_table_file = snakemake.input["edge_table_file"]
    parameters = snakemake.params["parameters"]
    output_train_net_file = snakemake.output["output_train_net_file"]
    output_target_edge_table_file = snakemake.output["output_target_edge_table_file"]
else:
    input_file = "../data/"
    output_file = "../data/"

# =================
# Load
# =================
edge_table = pd.read_csv(edge_table_file)

# =====================
# Construct the network
# =====================
src, trg = edge_table["src"].values, edge_table["trg"].values
n_nodes = int(np.maximum(np.max(src), np.max(trg)) + 1)
net = sparse.csr_matrix((np.ones_like(src), (src, trg)), shape=(n_nodes, n_nodes))

# =======================================
# Generate the link prediction benchmark
# =======================================
model = LinkPredictionDataset(
    testEdgeFraction=parameters["testEdgeFraction"],
    negative_edge_sampler=parameters["negativeEdgeSampler"],
)

model.fit(net)
train_net, target_edge_table = model.transform()


# ===============
# Save
# ===============
sparse.save_npz(output_train_net_file, train_net)
target_edge_table.to_csv(output_target_edge_table_file, index=False)
