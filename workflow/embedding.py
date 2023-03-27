# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 15:08:01
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-03-27 18:33:47
import sys
import embcom
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

#
# Input
#
if "snakemake" in sys.modules:
    netfile = snakemake.input["net_file"]
    embfile = snakemake.output["output_file"]
    params = snakemake.params["parameters"]
else:
    netfile = "../../data/multi_partition_model/networks/net_n~100000_K~2_cave~10_mu~0.10_sample~0.npz"
    embfile = "tmp.npz"
    dim = 64
    window_length = 10
    model_name = "torch-modularity"
    num_walks = 40

dim = int(params["dim"])
window_length = 10
model_name = params["model"]
num_walks = 40

net = sparse.load_npz(netfile)
net = net + net.T
net.data = net.data * 0.0 + 1.0

#
# Embedding models
#
if model_name == "node2vec":
    model = embcom.embeddings.Node2Vec(window_length=window_length, num_walks=num_walks)
elif model_name == "depthfirst-node2vec":
    model = embcom.embeddings.Node2Vec(
        window_length=window_length, num_walks=num_walks, p=100, q=1
    )
elif model_name == "deepwalk":
    model = embcom.embeddings.DeepWalk(window_length=window_length, num_walks=num_walks)
elif model_name == "line":
    model = embcom.embeddings.Node2Vec(window_length=1, num_walks=num_walks, p=1, q=1)
elif model_name == "glove":
    model = embcom.embeddings.Glove(window_length=window_length, num_walks=num_walks)
elif model_name == "leigenmap":
    model = embcom.embeddings.LaplacianEigenMap()
elif model_name == "adjspec":
    model = embcom.embeddings.AdjacencySpectralEmbedding()
elif model_name == "modspec":
    model = embcom.embeddings.ModularitySpectralEmbedding()
elif model_name == "nonbacktracking":
    model = embcom.embeddings.NonBacktrackingSpectralEmbedding()
elif model_name == "degree":
    model = embcom.embeddings.DegreeEmbedding()

#
# Embedding
#
# Get the largest connected component
net = sparse.csr_matrix(net)
component_ids = connected_components(net)[1]
u_component_ids, freq = np.unique(component_ids, return_counts=True)
ids = np.where(u_component_ids[np.argmax(freq)] == component_ids)[0]
H = sparse.csr_matrix(
    (np.ones_like(ids), (ids, np.arange(len(ids)))), shape=(net.shape[0], len(ids))
)
HT = sparse.csr_matrix(H.T)
net_ = HT @ net @ H
model.fit(net_)
emb_ = model.transform(dim=dim)

# Enlarge the embedding to the size of the original net
# All nodes that do not belong to the largest connected component have nan
ids = np.where(u_component_ids[np.argmax(freq)] != component_ids)[0]
emb = H @ emb_
emb[ids, :] = np.nan

# %%
#
# Save
#
np.savez_compressed(
    embfile,
    emb=emb,
    window_length=window_length,
    dim=dim,
    model_name=model_name,
)
