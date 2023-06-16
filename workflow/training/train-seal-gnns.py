# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 15:08:01
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-16 10:51:37
# %%
import sys
import numpy as np
from scipy import sparse
from seal import gnns, node_samplers
from seal.seal import SEAL, train
import torch
import torch_geometric

#
# Input
#
if "snakemake" in sys.modules:
    netfile = snakemake.input["net_file"]
    embfile = snakemake.output["output_file"]
    params = snakemake.params["parameters"]
else:
    netfile = "../data/derived/datasets/astro-ph/train-net_testEdgeFraction~0.5_sampleId~0.npz"
    embfile = "tmp.npz"
    params = {"model": "GCN", "dim": 64}

dim = int(params["dim"])
model_name = params["model"]

net = sparse.load_npz(netfile)
net = net + net.T
net.data = net.data * 0.0 + 1.0

#
# Embedding
#
# Get the largest connected component
net = sparse.csr_matrix(net)
import embcom

model = embcom.embeddings.SpectralGraphTransformation(
    kernel_func="exp", kernel_matrix="normalized_A"
)
model.fit(net)
model


class EmbeddingLinkPredictor:
    def __init__(self, emb):
        self.emb = emb

    def predict(self, src, trg):
        return self.emb[src]


model = EmbeddingLinkPredictor(model.transform(dim=64))
# %%
import pickle

with open("tmp.pickle", "wb") as f:
    pickle.dump(model, f)

with open("tmp.pickle", "rb") as f:
    model2 = pickle.load(f)
model2.predict(1, 1)
# %%
# emb = model.transform(dim=dim)

# Embedding
feature_vec = gnns.generate_base_embedding(net, 64)
feature_vec = torch.FloatTensor(feature_vec)
feature_dim = feature_vec.shape[1] + 1
dim_h = 64
dim = 64
gnn_model = torch_geometric.nn.models.GCN(
    in_channels=feature_dim,
    hidden_channels=dim_h,
    num_layers=2,
    out_channels=dim,
)

model = SEAL(gnn_model=gnn_model)
model = train(
    model=model,
    feature_vec=feature_vec,
    net=net,
    device="cuda:7",
    epochs=10,
    hops=2,
    feature_vec_dim=64,
    negative_edge_sampler=node_samplers.degreeBiasedNegativeEdgeSampling,
    # negative_edge_sampler=negative_uniform,
    batch_size=50,
    lr=0.01,
)
model.to("cpu")
emb = embedding_models[model_name](net, dim=dim)

# %%
#
# Save
#
np.savez_compressed(
    embfile,
    emb=emb,
    dim=dim,
    model_name=model_name,
)
