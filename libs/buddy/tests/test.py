# %%
"""
Test script for BUDDY model using Zachary's Karate Club network
"""
# %%
import numpy as np
from scipy import sparse
import networkx as nx
import torch
import argparse
from pathlib import Path
import buddy


def create_karate_club_data():
    """
    Create Karate Club network with random Gaussian features
    """
    # Create Karate Club network
    G = nx.karate_club_graph()
    num_nodes = G.number_of_nodes()

    # Create adjacency matrix
    adj_matrix = nx.to_scipy_sparse_array(G)

    # Create random Gaussian features (num_nodes x 6)
    np.random.seed(42)  # For reproducibility
    node_features = np.random.normal(0, 1, (num_nodes, 6))

    # Create test edges (20% of existing edges)
    edges = list(G.edges())
    np.random.shuffle(edges)
    num_test = int(0.2 * len(edges))
    test_edges = np.array(edges[:num_test])

    # Create negative test edges
    non_edges = list(nx.non_edges(G))
    np.random.shuffle(non_edges)
    test_edges_neg = np.array(non_edges[:num_test])

    # Remove test edges from adjacency matrix
    for edge in test_edges:
        adj_matrix[edge[0], edge[1]] = 0
        adj_matrix[edge[1], edge[0]] = 0  # Since undirected

    return adj_matrix, node_features, test_edges, test_edges_neg


from math import inf
from buddy.utils import ROOT_DIR, print_model_params, select_embedding, str2bool
import networkx as nx
from argparse import Namespace

# Create example data
adj_matrix, node_features, test_edges, test_edges_neg = create_karate_club_data()
node_features = None
config = buddy.BuddyConfig()
config.use_feature = False
# Run model
model, config = buddy.train_heldout(
    adj_matrix,
    model_file_path="saved_models",
    config=config,
#    param_ranges={
#        "num_hops": [1, 2],
#        "hidden_channels": [256, 1024],
#        "feature_dropout": [0.2, 0.5],
#        "use_RA": [True, False],
#    },
    max_patience=2,
    device="cuda:1",
)
# %%
# config = model.get_config()
config.use_RA, config.feature_dropout, config.hidden_channels, config.num_hops
# %%
model, config = buddy.load_model(model_path="saved_models", device="cuda:1")

# %%
model.get_config()
# %%
import networkx as nx

G = nx.karate_club_graph()
A = nx.adjacency_matrix(G)
labels = np.unique([d[1]["club"] for d in G.nodes(data=True)], return_inverse=True)[1]

emb = model.node_embedding.weight.data.detach().cpu().numpy()
# %%
nemb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
from sklearn.decomposition import PCA

xy = PCA(n_components=2).fit_transform(emb)
import matplotlib.pyplot as plt

plt.scatter(xy[:, 0], xy[:, 1], c=labels)
plt.show()
# %%
src_nodes, dst_nodes = np.triu_indices(adj_matrix.shape[0], k=1)
candidate_edges = torch.from_numpy(np.column_stack([src_nodes, dst_nodes])).long().T

preds = buddy.predict_edge_likelihood(
    model.to("cpu"),
    adj_matrix=adj_matrix,
    candidate_edges=candidate_edges,
    args=config,
    device="cuda:1",
)
import seaborn as sns

ped_mat = np.zeros((adj_matrix.shape[0], adj_matrix.shape[0]))
src_nodes, dst_nodes = tuple(candidate_edges.t().cpu().numpy().T)
ped_mat[src_nodes, dst_nodes] = preds
sns.heatmap(ped_mat + ped_mat.T)

# %%
preds.numpy()

# %%
