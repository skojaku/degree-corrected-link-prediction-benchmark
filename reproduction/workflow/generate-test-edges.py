# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-01-17 03:57:05
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-04 05:43:02
# %%
import numpy as np
import pandas as pd
from scipy import sparse
import sys
from tqdm import tqdm
from gnn_tools.LinkPredictionDataset import NegativeEdgeSampler
import utils

if "snakemake" in sys.modules:
    edge_table_file = snakemake.input["edge_table_file"]
    parameters = snakemake.params["parameters"]
    train_net_file = snakemake.input["train_net_file"]
    output_target_edge_table_file = snakemake.output["output_target_edge_table_file"]
else:
    data = "wiki-Vote"
    edge_table_file = f"../data/derived/networks/preprocessed/{data}/edge_table.csv"
    parameters = {"negativeEdgeSampler": "uniform"}
    train_net_file = (
        f"../data/derived/datasets/{data}/train-net_testEdgeFraction~0.5_sampleId~1.npz"
    )
    output_file = "../data/"

# =================
# Load
# =================
edge_table = pd.read_csv(edge_table_file)
train_net = sparse.load_npz(train_net_file)

# =====================
# Construct the network
# =====================
src, trg = edge_table["src"].values, edge_table["trg"].values
n_nodes = int(np.maximum(np.max(src), np.max(trg)) + 1)
net = utils.edgeList2adjacencyMatrix(src, trg, n_nodes)
train_net = utils.to_undirected(train_net)

# =====================
# Find the test edges
# =====================
train_src, train_trg, _ = sparse.find(train_net)
test_edge_indices = np.array(
    list(
        set(utils.pairing(src, trg)).difference(
            set(utils.pairing(train_src, train_trg))
        )
    )
)
test_src, test_trg = utils.depairing(test_edge_indices)

# ===========================
# Generate the negative edges
# ===========================
model = NegativeEdgeSampler(negative_edge_sampler=parameters["negativeEdgeSampler"])
model.fit(net)
test_src, test_trg = np.concatenate([test_src, test_trg]), np.concatenate(
    [test_trg, test_src]
)
neg_src, neg_trg = model.sampling(source_nodes=test_src, size=len(test_src))

# ===============
# Save
# ===============
target_edge_table = pd.DataFrame(
    {
        "src": np.concatenate([test_src, neg_src]),
        "trg": np.concatenate([test_trg, neg_trg]),
        "isPositiveEdge": np.concatenate(
            [np.ones_like(test_src), np.zeros_like(neg_trg)]
        ),
    }
)

target_edge_table.to_csv(output_target_edge_table_file, index=False)
