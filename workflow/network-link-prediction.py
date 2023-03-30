# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-01-17 04:25:23
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-03-30 10:35:18
# %%
from scipy import sparse
import numpy as np
import pandas as pd
import sys

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    net_file = snakemake.input["net_file"]
    params = snakemake.params["parameters"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/"
    output_file = "../data/"

# ========================
# Load
# ========================
edge_table = pd.read_csv(input_file)
net = sparse.load_npz(net_file)
model = params["model"]

# To ensure that the network is undirected and unweighted.
net = net + net.T
net.data = net.data * 0.0 + 1.0

# ========================
# Preprocess
# ========================
def calc_topo_link_pred_score(r, c, net, predictor):

    # Degree product
    deg = np.array(net.sum(axis=1)).reshape(-1)

    if predictor == "preferentialAttachment":
        return deg[r] * deg[c]
    elif predictor == "commonNeighbors":
        score = np.array((net[r, :].multiply(net[c, :])).sum(axis=1)).reshape(-1)
        return score
    elif predictor == "jaccardIndex":
        score = np.array((net[r, :].multiply(net[c, :])).sum(axis=1)).reshape(-1)
        score = score / np.maximum(deg[r] + deg[c] - score, 1)
        return score
    elif predictor == "resourceAllocation":
        deg_inv = 1 / np.maximum(deg, 1)
        deg_inv[deg == 0] = 0
        score = np.array(
            ((net[r, :] @ sparse.diags(deg_inv)).multiply(net[c, :])).sum(axis=1)
        ).reshape(-1)
        return score
    elif predictor == "adamicAdar":
        log_deg_inv = 1 / np.maximum(np.log(np.maximum(deg, 1)), 1)
        log_deg_inv[deg == 0] = 0
        score = np.array(
            ((net[r, :] @ sparse.diags(log_deg_inv)).multiply(net[c, :])).sum(axis=1)
        ).reshape(-1)
        return score


src, trg, y = (
    edge_table["src"].values.astype(int),
    edge_table["trg"].values.astype(int),
    edge_table["isPositiveEdge"],
)


ypred = calc_topo_link_pred_score(src, trg, net, model)

# ========================
# Save
# ========================
pd.DataFrame({"y": y, "ypred": ypred}).to_csv(output_file)
