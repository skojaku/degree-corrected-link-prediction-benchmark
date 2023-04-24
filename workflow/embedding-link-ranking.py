# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-01-17 04:25:23
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-04-01 07:12:09
# %%
import numpy as np
import pandas as pd
import sys
from RankingModels import *

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    emb_file = snakemake.input["emb_file"]
    net_file = snakemake.input["net_file"]
    topK = int(snakemake.params["topK"])
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/"
    output_file = "../data/"

# ========================
# Load
# ========================
edge_table = pd.read_csv(input_file)
emb = np.load(emb_file)["emb"]
net = sparse.load_npz(net_file)

# ========================
# Preprocess
# ========================
net = net + net.T
net.data = net.data * 0.0 + 1.0

R = ranking_by_embedding(emb, max_k=topK)

R = R - R.multiply(net)  # remove positive edges from the ranking
R.eliminate_zeros()
R.sort_indices()

# ========================
# Save
# ========================
sparse.save_npz(output_file, R)
