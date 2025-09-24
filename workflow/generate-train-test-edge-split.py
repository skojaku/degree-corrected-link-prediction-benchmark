# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-01-17 03:57:05
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-04-12 17:37:44
# %%
import numpy as np
import pandas as pd
from scipy import sparse
import sys
from tqdm import tqdm
from gnn_tools.LinkPredictionDataset import TrainTestEdgeSplitter
import utils

if "snakemake" in sys.modules:
    edge_table_file = snakemake.input["edge_table_file"]
    parameters = snakemake.params["parameters"]
    output_train_net_file = snakemake.output["output_train_net_file"]
    output_test_edge_file = snakemake.output["output_test_edge_file"]
else:
    edge_table_file = "../data/preprocessed/ogbl-collab/edge_table.csv"
    output_file = "../data/"

# =================
# Load
# =================
edge_table = pd.read_csv(edge_table_file)

# %%
# =====================
# Construct the network
# =====================
src, trg = edge_table["src"].values, edge_table["trg"].values
n_nodes = int(np.maximum(np.max(src), np.max(trg)) + 1)
net = utils.edgeList2adjacencyMatrix(src, trg, n_nodes)

# =======================================
# Generate the link prediction benchmark
# =======================================
model = TrainTestEdgeSplitter(fraction=parameters["testEdgeFraction"])
model.fit(net)
test_src, test_trg = model.test_edges_
train_src, train_trg = model.train_edges_
train_net = utils.edgeList2adjacencyMatrix(train_src, train_trg, n_nodes)

# ===============
# Save
# ===============
sparse.save_npz(output_train_net_file, train_net)
pd.DataFrame(
    {"src": test_src, "trg": test_trg, "isPositiveEdge": np.ones_like(test_src)}
).to_csv(output_test_edge_file, index=False)
