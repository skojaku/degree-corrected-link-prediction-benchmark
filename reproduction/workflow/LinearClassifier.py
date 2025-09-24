# %%
import torch
import numpy as np
from tqdm import tqdm
from scipy import sparse
from sklearn.metrics import roc_auc_score
from gnn_tools.LinkPredictionDataset import NegativeEdgeSampler
from NetworkTopologyPredictionModels import (
    resourceAllocation,
    commonNeighbors,
    jaccardIndex,
    adamicAdar,
    localRandomWalk,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle


class LinearClassifier:
    def __init__(self, negative_edge_sampler="uniform"):
        self.negative_edge_sampler = negative_edge_sampler
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model = LogisticRegression()
        self.model.fit(X_scaled, y)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, network, src, trg):
        X = compute_network_stats(network, src, trg, with_degree=True)
        return self.predict_proba(X)

    def train(self, network):
        pos_src, pos_trg, _ = sparse.find(sparse.triu(network, 1))
        sampler = NegativeEdgeSampler(negative_edge_sampler=self.negative_edge_sampler)
        sampler.fit(network)
        neg_src, neg_trg = sampler.sampling(source_nodes=pos_src, size=len(pos_src))

        X = compute_network_stats(
            network,
            np.concatenate([pos_src, neg_src]),
            np.concatenate([pos_trg, neg_trg]),
            with_degree=True,
        )
        y = np.concatenate([np.ones(len(pos_src)), np.zeros(len(neg_src))])
        order = np.random.permutation(len(X))
        X = X[order]
        y = y[order]
        self.fit(X, y)

    def save(self, filepath):
        """Save the trained model and scaling parameters to disk.

        Args:
            filepath (str): Path where model should be saved
        """
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "scaler": self.scaler,
                    "negative_edge_sampler": self.negative_edge_sampler,
                },
                f,
            )

    def load(self, filepath):
        """Load a trained model from disk.

        Args:
            filepath (str): Path to saved model file
        """
        with open(filepath, "rb") as f:
            checkpoint = pickle.load(f)
            self.model = checkpoint["model"]
            self.scaler = checkpoint["scaler"]
            self.negative_edge_sampler = checkpoint["negative_edge_sampler"]


def compute_network_stats(network, src, trg, with_degree=False, std=None, mean=None):
    """Compute network statistics used as features for prediction.

    Args:
        network: scipy sparse matrix representing the network
        src: source node indices (optional)
        trg: target node indices (optional)
        maxk: number of top predictions to return (optional)

    Returns:
        Array of network statistics for each node pair
    """

    # Compute each network statistic
    # Each score is a numpy array of shape (len(src),)
    ra = resourceAllocation(network, src, trg, maxk=None)
    # cn = commonNeighbors(network, src, trg, maxk=None)
    ji = jaccardIndex(network, src, trg, maxk=None)
    aa = adamicAdar(network, src, trg, maxk=None)
    lrw = localRandomWalk(network, src, trg, maxk=None)

    if with_degree:
        # Add degree product if enabled
        deg = network.sum(axis=1).A1
        features = [
            ra,
            ji,
            aa,
            lrw,
            np.minimum(deg[trg], deg[src]),
            np.maximum(deg[trg], deg[src]),
            deg[src] * deg[trg],
        ]
    else:
        features = [ra, ji, aa, lrw]

    X = np.column_stack(features)
    if std is not None and mean is not None:
        X = (X - mean) / std
    return X


# import networkx as nx
#
# G = nx.karate_club_graph()
# A = nx.adjacency_matrix(G)
# labels = np.unique([d[1]["club"] for d in G.nodes(data=True)], return_inverse=True)[1]
#
## Test script for MLP training
# A = sparse.csr_matrix(nx.adjacency_matrix(G))
#
## Create train/test split
# from gnn_tools.LinkPredictionDataset import TrainTestEdgeSplitter
#
# splitter = TrainTestEdgeSplitter(fraction=0.2)
# splitter.fit(A)
# train_src, train_trg = splitter.train_edges_
# test_src, test_trg = splitter.test_edges_
#
# train_net = sparse.csr_matrix(
#    (np.ones(len(train_src)), (train_src, train_trg)), shape=A.shape
# )
#
## Generate negative test edges
# sampler = NegativeEdgeSampler(negative_edge_sampler="uniform")
# sampler.fit(A)
# neg_src, neg_trg = sampler.sampling(source_nodes=test_src, size=len(test_src))
#
## Combine positive and negative edges
# test_src = np.concatenate([test_src, neg_src])
# test_trg = np.concatenate([test_trg, neg_trg])
# test_labels = np.concatenate(
#    [np.ones(len(splitter.test_edges_[0])), np.zeros(len(neg_src))]
# )
#
## Train MLP model
#
# model = LinearClassifier()
# model.train(train_net)
# preds = model.predict(train_net, test_src, test_trg)
# auc_score = roc_auc_score(test_labels, preds)
# print(f"Test AUC: {auc_score:.4f}")
# model.model.coef_
## %%
#
