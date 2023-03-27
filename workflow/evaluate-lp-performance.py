# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-01-17 04:25:23
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-03-27 18:29:00
# %%
import numpy as np
import pandas as pd
from scipy import sparse
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

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

emb[pd.isna(emb)] = 0

# ========================
# Preprocess
# ========================

src, trg, isPositiveEdge = (
    edge_table["src"].values.astype(int),
    edge_table["trg"].values.astype(int),
    edge_table["isPositiveEdge"],
)

s = isPositiveEdge > 0
pos_edges = (src[s], trg[s])

s = isPositiveEdge == 0
neg_edges = (src[s], trg[s])
from sklearn.metrics import roc_auc_score

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
score = roc_auc_score(y_true=y, y_score=ypred)


# ========================
# Save
# ========================

pd.DataFrame({"score": [score]}).to_csv(output_file)
