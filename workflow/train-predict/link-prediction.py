# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-07-03 13:21:14
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-07-31 15:49:51
# %%
import itertools
import numpy as np
import pandas as pd
from scipy import sparse
import sys
import GPUtil
from models.LinkPredictionModel import link_prediction_models

if "snakemake" in sys.modules:
    train_net_file = snakemake.input["train_net_file"]
    model_file = snakemake.input["model_file"]
    params = snakemake.params["parameters"]
    output_file = snakemake.output["output_file"]
else:
    train_net_file = "../../mydata/derived/datasets/bitcoinalpha/train-net_trainTestSplit~8d836754_testEdgeFraction~0.5.npz"
    model_file = "../../mydata/derived/models/bitcoinalpha/model_trainTestSplit~8d836754_testEdgeFraction~0.5_PredictionModel~9b3de853_model~seal+GCN_modelType~seal.pickle"
    params = {
        "model": "seal+GCN",
        "feature_dim": 64,
        "dim_h": 64,
        "num_layers": 2,
        "negative_edge_sampler": "uniform",
        "epochs": 10,
        "hops": 2,
        "batch_size": 50,
        "lr": 1e-3,
        "in_channels": 64,
        "hidden_channels": 64,
        "num_layers": 2,
        "out_channels": 64,
        "modelType": "seal",
    }

# ========================
# Load
# ========================
train_net = sparse.load_npz(train_net_file)

model = link_prediction_models[params["modelType"]](**params)
model.load(model_file)


# if "device" not in params:
#    device = GPUtil.getAvailable(
#        order="random",
#        limit=99,
#        maxMemory=0.5,
#        maxLoad=0.5,
#        # excludeID=[0, 5],
#    )[0]
#    device = f"cuda:{device}"
#    params["device"] = device

# ========================
# Preprocess
# ========================
n_nodes = train_net.shape[0]


# Find the edges to be tested
def pairing(r, c):
    r, c = r.astype(int), c.astype(int)
    return np.minimum(r, c) + 1j * np.maximum(r, c)


def depairing(v):
    return np.real(v).astype(int), np.imag(v).astype(int)


src, trg, _ = sparse.find(train_net)
edges = set(pairing(src, trg))

# step = 2
A = train_net.copy()
A = A @ A
src, trg, _ = sparse.find(sparse.triu(A, k=1))
new_edges = set(pairing(src, trg)).difference(edges)
new_src, new_trg = depairing(list(new_edges))
n_samples = len(new_src)

edge_table = []
edge_table.append(pd.DataFrame({"src": new_src, "trg": new_trg, "path_length": 2}))
edges.update(new_edges)

# step = 3
A = train_net @ A
src, trg, _ = sparse.find(sparse.triu(A, k=1))
new_edges = set(pairing(src, trg)).difference(edges)
new_src, new_trg = depairing(list(new_edges))

if len(new_src) > n_samples:
    np.random.seed(seed=42)
    s = np.random.choice(len(new_src), size=n_samples, replace=False)
    new_src, new_trg = new_src[s], new_trg[s]

edge_table.append(pd.DataFrame({"src": new_src, "trg": new_trg, "path_length": 3}))
edges.update(new_edges)

edge_table = pd.concat(edge_table)

# ========================
# Link prediction
# ========================
test_src, test_trg = tuple(edge_table[["src", "trg"]].values.T)
score = model.predict(train_net, test_src, test_trg, **params)

# ========================
# Save
# ========================
path_length = edge_table["path_length"].values
np.savez(output_file, src=test_src, trg=test_trg, score=score, path_length=path_length)

# %%
