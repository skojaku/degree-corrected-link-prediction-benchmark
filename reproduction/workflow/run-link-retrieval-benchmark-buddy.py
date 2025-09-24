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
import sys


#  Load
if "snakemake" in sys.modules:
    train_net_file = snakemake.input["train_net_file"]
    test_edge_file = snakemake.input["test_edge_file"]
    model_file = snakemake.input["model_file"]
    data_name = snakemake.params["data_name"]
    output_file = snakemake.output["output_file"]
else:
    data_name = "airport-rach"
    data_name = "maayan-foodweb"
    train_net_file = f"../data/derived/datasets/{data_name}/train-net_testEdgeFraction~0.25_sampleId~3.npz"
    test_edge_file = f"../data/derived/datasets/{data_name}/testEdgeTable_testEdgeFraction~0.25_sampleId~3.csv"
    output_file = ""
    maxk = 100
    model = "commonNeighbors"
    model_type = "embedding"
    model_file = f"../data/derived/models/buddy/{data_name}/buddy_model~Buddy_testEdgeFraction~0.25_sampleId~3"
    model = "localRandomWalk"
    # model_type = "topology"
    # sampling = "uniform"
    # sampling = "degreeBiased"

def get_gpu_id(excludeID=[]):
    device = GPUtil.getFirstAvailable(
        order="random",
        maxLoad=1.0,
        maxMemory=0.6,
        attempts=99999,
        interval=60 * 1,
        verbose=False,
        # excludeID=excludeID,
        # excludeID=[6, 7],
    )[0]
    device = f"cuda:{device}"
    return device

device = get_gpu_id()

#  Preprocess
train_net = sparse.load_npz(train_net_file)
train_net = train_net + train_net.T
train_net.data = train_net.data * 0.0 + 1.0
train_net.shape

# %%

df = pd.read_csv(test_edge_file)
test_net = sparse.csr_matrix(
    (df["isPositiveEdge"], (df["src"], df["trg"])), shape=train_net.shape
)

maxk = 100
maxk = np.minimum(maxk, train_net.shape[1] - 1)

model, config = buddy.load_model(model_path=model_file, device="cpu")

scores, indices = local_random_walk_forward_push(train_net, maxk = maxk * 2)

src = np.arange(train_net.shape[0]).reshape((-1,1)) @ np.ones((1, indices.shape[1]))
trg = indices
src, trg = src.flatten(), trg.flatten()
candidate_edges = torch.from_numpy(np.column_stack([src, trg])).long().T

preds = buddy.predict_edge_likelihood(
    model, train_net, candidate_edges, args=config, device=device
)
preds = preds.numpy()
predicted = sparse.csr_matrix((preds, (src, trg)), shape=train_net.shape)

# %%
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

# %%
