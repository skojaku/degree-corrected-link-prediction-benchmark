# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 15:08:01
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-03-31 17:19:09
# %%
import sys
import numpy as np
from scipy import sparse
from EmbeddingModels import *

#
# Input
#
if "snakemake" in sys.modules:
    netfile = snakemake.input["net_file"]
    embfile = snakemake.output["output_file"]
    params = snakemake.params["parameters"]
else:
    netfile = "../data/derived/datasets/astro-ph/train-net_negativeEdgeSampler~degreeBiased_testEdgeFraction~0.5_sampleId~0.npz"
    embfile = "tmp.npz"
    params = {"model": "node2vec", "dim": 64}

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

# Embedding
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
