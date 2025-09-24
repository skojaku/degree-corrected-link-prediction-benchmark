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
import faiss
import sys
from sklearn.linear_model import LogisticRegression
import scipy
import GPUtil

# %% Load
if "snakemake" in sys.modules:
    train_net_file = snakemake.input["train_net_file"]
    test_edge_file = snakemake.input["test_edge_file"]
    emb_file = (
        snakemake.input["emb_file"]
        if "emb_file" in list(snakemake.input.keys())
        else None
    )
    model = snakemake.params["model"]
    model_type = snakemake.params["model_type"]
    # sampling = snakemake.params["sampling"]
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


def sample_candidates_faiss(
    emb,
    net,
    maxk=100,
    sampling="uniform",
    test_edge_frac=0.25,
):
    res = faiss.StandardGpuResources()  # Use standard GPU resources
    gpu_id = GPUtil.getFirstAvailable(
        order="memory",
        maxLoad=1,
        maxMemory=0.5,
        attempts=99999,
        interval=60 * 1,
        verbose=False,
    )[0]

    if sampling == "degreeBiased":
        index = faiss.IndexFlatIP(emb.shape[1] + 1)  # Initialize the index on CPU
        gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, index)  # Move the index to GPU
        deg = np.array(net.sum(axis=1)).flatten()
        emb_key, emb_query = emb.copy(), emb.copy()
        emb_key = np.hstack([emb_key, np.log(np.maximum(1, deg.reshape(-1, 1)))])
        emb_query = np.hstack([emb_query, np.ones((emb_query.shape[0], 1))])
        gpu_index.add(emb_key.astype("float32"))  # Add vectors to the index on GPU
        D, I = gpu_index.search(
            emb_query.astype("float32"), int(maxk)
        )  # Perform the search on GPU
        return D, I

    index = faiss.IndexFlatIP(emb.shape[1])  # Initialize the index on CPU
    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, index)  # Move the index to GPU
    gpu_index.add(emb.astype("float32"))  # Add vectors to the index on GPU
    D, I = gpu_index.search(
        emb.astype("float32"), int(maxk)
    )  # Perform the search on GPU
    return D, I


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

if model_type == "topology":
    scores, predicted = topology_models[model](train_net, maxk=maxk)
elif model_type == "embedding":
    assert emb_file is not None, "embedding file must be provided"
    data = np.load(emb_file)
    emb = data["emb"]
    scores, predicted = sample_candidates_faiss(
        emb,
        train_net,
        maxk=maxk,
        sampling=(
            "uniform"
            # if model not in ["dcGIN", "dcGAT", "dcGraphSAGE", "dcGCN"]
            # else "degreeBiased"
        ),
    )
#
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
