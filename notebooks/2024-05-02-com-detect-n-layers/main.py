# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 15:08:01
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-10 08:46:51
# %%
import sys
import numpy as np
from scipy import sparse
import pandas as pd
from gnn_tools.models import *

#
# Input
#
if "snakemake" in sys.modules:
    netfile = snakemake.input["net_file"]
    embfile = snakemake.output["output_file"]
    params = snakemake.params["parameters"]
else:
    netfile = "../../data/derived/community-detection-datasets/lfr/networks/net_n~3000_k~25_tau~3_tau2~3_minc~100_maxk~1000_maxc~1000_mu~0.50_sample~2.npz"
    nodefile = "../../data/derived/community-detection-datasets/lfr/networks/node_n~3000_k~25_tau~3_tau2~3_minc~100_maxk~1000_maxc~1000_mu~0.50_sample~2.npz"
    embfile = "tmp.npz"
    params = {"model": "GraphSAGE", "dim": 128}

dim = int(params["dim"])
model_name = params["model"]

net = sparse.load_npz(netfile)
net = net + net.T
net.data = net.data * 0.0 + 1.0

node_table = pd.read_csv(nodefile)
y = node_table["membership"].values
# %%
#
# Embedding
#
# Get the largest connected component
net = sparse.csr_matrix(net)

# %%
deg = net.sum(axis=1).A1
np.max(deg)
# %% Embedding
model_name = "dcGAT"
dim = 64
dim = np.minimum(dim, net.shape[0] - 5)
emb = embedding_models[model_name](
    net,
    dim=dim,
    lr=1e-2,
    epochs=250,
    clustering="modularity",
    # memberships=y,
    batch_size=5000,
    dropout=0,
)
# %%
from sklearn.cluster import KMeans

X = emb.copy()
X = np.einsum("ij,i->ij", X, 1.0 / np.linalg.norm(X, axis=1))

K = np.max(y) + 1
kmeans = KMeans(n_clusters=K, random_state=0).fit(emb)
pred = kmeans.labels_


from sklearn.metrics import normalized_mutual_info_score

print(normalized_mutual_info_score(y, pred))


# %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis(n_components=2)
clf.fit(emb, y)
print(clf.score(emb, y))

# %%
from sklearn.decomposition import PCA
import seaborn as sns

xy = PCA(n_components=2).fit_transform(emb)
sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=y)

# %%
