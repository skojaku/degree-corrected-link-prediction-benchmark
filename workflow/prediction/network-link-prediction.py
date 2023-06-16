# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-01-17 04:25:23
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-04-11 17:47:04
from scipy import sparse
import numpy as np
import pandas as pd
import sys
from NetworkTopologyPredictionModels import *

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    net_file = snakemake.input["net_file"]
    params = snakemake.params["parameters"]
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

src, trg, y = (
    edge_table["src"].values.astype(int),
    edge_table["trg"].values.astype(int),
    edge_table["isPositiveEdge"],
)


ypred = topology_models[model](net, src, trg)

# ========================
# Save
# ========================
pd.DataFrame({"y": y, "ypred": ypred}).to_csv(output_file)
