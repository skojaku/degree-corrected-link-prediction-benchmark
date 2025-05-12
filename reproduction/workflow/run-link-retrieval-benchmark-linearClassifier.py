# %% Load package
import numpy as np
import pandas as pd
from scipy import sparse
import sys
from NetworkTopologyPredictionModels import *
import torch
import torch.nn as nn
import torch.optim as optim
import gnn_tools
import sys
from sklearn.linear_model import LogisticRegression
import scipy
import GPUtil
import buddy

# %% Load
if "snakemake" in sys.modules:
    train_net_file = snakemake.input["train_net_file"]
    test_edge_file = snakemake.input["test_edge_file"]
    model_file = snakemake.input["model_file"]
    data_name = snakemake.params["data_name"]
    output_file = snakemake.output["output_file"]
else:
    data_name = "airport-rach"
    data_name = "subelj-cora-cora"
    train_net_file = f"../data/derived/datasets/{data_name}/train-net_testEdgeFraction~0.25_sampleId~0.npz"
    test_edge_file = f"../data/derived/datasets/{data_name}/testEdgeTable_testEdgeFraction~0.25_sampleId~0.csv"
    output_file = "."
    maxk = 100
    model = "commonNeighbors"
    model_type = "embedding"
    emb_file = f"../data/derived/embedding/{data_name}/emb_testEdgeFraction~0.25_sampleId~0_model~node2vec_dim~128.npz"
    # emb_file = f"../data/derived/embedding/{data_name}/emb_testEdgeFraction~0.25_sampleId~0_model~dcGCN_dim~128.npz"
    # emb_file = f"../data/derived/embedding/{data_name}/emb_testEdgeFraction~0.25_sampleId~0_model~GCN_dim~128.npz"
    emb_file = f"../data/derived/embedding/{data_name}/emb_testEdgeFraction~0.25_sampleId~0_model~deepwalk_dim~128.npz"
    model = "localRandomWalk"
    # model_type = "topology"
    # sampling = "uniform"
    # sampling = "degreeBiased"


#  Preprocess
train_net = sparse.load_npz(train_net_file)
train_net = train_net + train_net.T
train_net.data = train_net.data * 0.0 + 1.0

df = pd.read_csv(test_edge_file)
test_net = sparse.csr_matrix(
    (df["isPositiveEdge"], (df["src"], df["trg"])), shape=train_net.shape
)

maxk = 100
maxk = np.minimum(maxk, train_net.shape[1] - 1)

S = train_net @ train_net
S = S - S.multiply(train_net)

src, trg, _ = sparse.find(S)

from LinearClassifier import LinearClassifier

model = LinearClassifier()
model.load(model_file)
preds = model.predict(train_net, src, trg)

predicted = sparse.csr_matrix((preds, (src, trg)), shape=train_net.shape)
scores, predicted = find_k_largest_elements(predicted, maxk)


# ========================
# Evaluations
# ========================


result = []
for topk in [5, 10, 25, 50, 100]:
    if topk > test_net.shape[1]:
        continue
    rec, prec, n_test = 0, 0, 0
    for i in range(test_net.shape[0]):
        if np.sum(test_net[i]) == 0:
            continue
        rec += np.sum(test_net[i, predicted[i, :topk]]) / np.sum(test_net[i])
        prec += np.sum(test_net[i, predicted[i, :topk]]) / topk
        n_test += 1
    rec /= n_test
    prec /= n_test
    result.append({"rec": rec, "prec": prec, "topk": topk, "data_name": data_name})

result = pd.DataFrame(result)

result.to_csv(output_file, index=False)
