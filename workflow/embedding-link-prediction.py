# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-01-17 04:25:23
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-04-19 17:58:33
# %%
import numpy as np
import pandas as pd
import sys
from scipy import sparse
from EmbeddingModels import calc_prob_i_j

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    emb_file = snakemake.input["emb_file"]
    net_file = snakemake.input["net_file"]
    model_name = snakemake.params["parameters"]["model"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/"
    output_file = "../data/"

# ========================
# Load
# ========================
edge_table = pd.read_csv(input_file)
net = sparse.load_npz(net_file)
emb = np.load(emb_file)["emb"]

# ========================
# Preprocess
# ========================

emb[pd.isna(emb)] = 0

src, trg, y = (
    edge_table["src"].values.astype(int),
    edge_table["trg"].values.astype(int),
    edge_table["isPositiveEdge"].astype(int),
)

ypred = calc_prob_i_j(emb, src, trg, net, model_name)

# ========================
# Save
# ========================
pd.DataFrame({"y": y, "ypred": ypred}).to_csv(output_file)
