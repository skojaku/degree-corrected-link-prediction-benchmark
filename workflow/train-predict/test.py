# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 15:08:01
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-07-28 17:30:00
# %%
import sys
import numpy as np
from scipy import sparse
import GPUtil
import torch
from sklearn.metrics import roc_auc_score

# sys.path.insert(0, "../..")
from models.LinkPredictionModel import link_prediction_models

#
# Input
#
if "snakemake" in sys.modules:
    netfile = snakemake.input["net_file"]
    output_trained_model_file = snakemake.output["output_file"]
    params = snakemake.params["parameters"]
else:
    netfile = "../../data/derived/datasets/celegans/train-net_trainTestSplit~8d836754_testEdgeFraction~0.5.npz"
    embfile = "tmp.npz"
    params = {
        "model": "seal+GCN",
        "feature_dim": 64,
        "dim_h": 64,
        "num_layers": 2,
        "dim": 64,
        "negative_edge_sampler": "uniform",
        "epochs": 10,
        "hops": 2,
        "batch_size": 50,
        "lr": 1e-3,
        "gnn_model": "GIN",
        "in_channels": 64,
        "hidden_channels": 64,
        "num_layers": 2,
        "out_channels": 64,
        "modelType": "seal",
        "device": "cuda:0",
    }


if "device" not in params:
    device = GPUtil.getAvailable(
        order="random",
        limit=99,
        maxMemory=0.5,
        maxLoad=0.5,
        excludeID=[6, 7],
    )[0]
    device = f"cuda:{device}"
    params["device"] = device

import pandas as pd

test_edge_table = pd.read_csv(
    "../../data/derived/datasets/celegans/targetEdgeTable_trainTestSplit~8d836754_testEdgeFraction~0.5_negativeSampling~c74d0456_negativeEdgeSampler~uniform.csv"
)
predictor = link_prediction_models[params["modelType"]](**params)

y = test_edge_table["isPositiveEdge"].values

net = sparse.load_npz(netfile)
net = net + net.T
net.data = net.data * 0.0 + 1.0

#
# Embedding
#
# Get the largest connected component
net = sparse.csr_matrix(net)

# Embedding
predictor.train(net)

# %%
# Save
#
# torch.save(predictor.state_dict(), output_trained_model_file)

test_src, test_trg = test_edge_table["src"].values, test_edge_table["trg"].values

# ========================
# Link prediction
# ========================
sz = np.sum(y)
r, c, v = sparse.find(net)
s = np.random.choice(len(v), size=sz, replace=False)
# test_src[y > 0] = r[s]
# test_trg[y > 0] = c[s]
score = predictor.predict(net, test_src, test_trg, batch_size=1)
roc_auc_score(y, score)
# roc_auc_score(test_edge_table["isPositiveEdge"], score)

# %%
