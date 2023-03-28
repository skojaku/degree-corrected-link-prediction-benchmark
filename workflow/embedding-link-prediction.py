# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-01-17 04:25:23
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-03-28 10:37:09
# %%
import numpy as np
import pandas as pd
import sys

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    emb_file = snakemake.input["emb_file"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/"
    output_file = "../data/"

# ========================
# Load
# ========================
edge_table = pd.read_csv(input_file)
emb = np.load(emb_file)["emb"]

# ========================
# Preprocess
# ========================

emb[pd.isna(emb)] = 0

src, trg, isPositiveEdge = (
    edge_table["src"].values.astype(int),
    edge_table["trg"].values.astype(int),
    edge_table["isPositiveEdge"],
)

s = isPositiveEdge > 0
pos_edges = (src[s], trg[s])

s = isPositiveEdge == 0
neg_edges = (src[s], trg[s])

pos_edges_src, pos_edges_trg = pos_edges
pos_score = np.array(
    np.sum(emb[pos_edges_src, :] * emb[pos_edges_trg, :], axis=1)
).reshape(-1)

neg_edges_src, neg_edges_trg = neg_edges
neg_score = np.array(
    np.sum(emb[neg_edges_src, :] * emb[neg_edges_trg, :], axis=1)
).reshape(-1)

y, ypred = np.concatenate(
    [np.ones_like(pos_score), np.zeros_like(neg_score)]
), np.concatenate([pos_score, neg_score])

# score = roc_auc_score(y_true=y, y_score=ypred)

# ========================
# Save
# ========================
pd.DataFrame({"y": y, "ypred": ypred}).to_csv(output_file)
