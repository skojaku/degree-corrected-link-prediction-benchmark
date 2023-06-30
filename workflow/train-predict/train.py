# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 15:08:01
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-30 18:26:07
# %%
import sys
import numpy as np
from scipy import sparse
import GPUtil

# sys.path.insert(0, "../..")
from models.EmbeddingModels import *
from models.StackingModel import *
from models.SEALModel import *
from models.NetworkModels import *

#
# Input
#
if "snakemake" in sys.modules:
    netfile = snakemake.input["net_file"]
    output_trained_model_file = snakemake.output["output_file"]
    params = snakemake.params["parameters"]
else:
    netfile = "../../data/derived/datasets/celegans/train-net_testEdgeFraction~0.5_sampleId~0.npz"
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
        order="first",
        limit=99,
        maxMemory=1,
        maxLoad=1,
        excludeID=[0, 5],
    )[0]
    device = f"cuda:{device}"
    params["device"] = device

predictor = {
    "embedding": EmbeddingLinkPredictor,
    "seal": SEALLinkPredictor,
    "stacklp": StackingLinkPredictor,
    "network": NetworkLinkPredictor,
}[params["modelType"]](**params)

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
#
# Save
#
torch.save(predictor.state_dict(), output_trained_model_file)
