# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-08-26 09:51:23
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-03-28 10:22:30
"""Module for embedding."""
# %%
import gensim
import networkx as nx
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GATv2Conv

from torch_geometric.utils import train_test_split_edges


from embcom import rsvd, samplers, utils, train


try:
    import glove
except ImportError:
    print(
        "Ignore this message if you do not use Glove. Otherwise, install glove python package by 'pip install glove_python_binary' "
    )


# Base class


class NodeEmbeddings:
    """Super class for node embedding class."""

    def __init__(self):
        self.in_vec = None
        self.out_vec = None

    def fit(self):
        """Estimating the parameters for embedding."""
        pass

    def transform(self, dim, return_out_vector=False):
        """Compute the coordinates of nodes in the embedding space of the
        prescribed dimensions."""
        # Update the in-vector and out-vector if
        # (i) this is the first to compute the vectors or
        # (ii) the dimension is different from that for the previous call of transform function
        if self.out_vec is None:
            self.update_embedding(dim)
        elif self.out_vec.shape[1] != dim:
            self.update_embedding(dim)
        return self.out_vec if return_out_vector else self.in_vec

    def update_embedding(self, dim):
        """Update embedding."""
        pass


class Node2Vec(NodeEmbeddings):
    """A python class for the node2vec.

    Parameters
    ----------
    num_walks : int (optional, default 10)
        Number of walks per node
    walk_length : int (optional, default 40)
        Length of walks
    window_length : int (optional, default 10)
        Restart probability of a random walker.
    p : node2vec parameter (TODO: Write doc)
    q : node2vec parameter (TODO: Write doc)
    """

    def __init__(
        self,
        num_walks=10,
        walk_length=80,
        window_length=10,
        p=1.0,
        q=1.0,
        verbose=False,
    ):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.window_length = window_length
        self.sampler = samplers.Node2VecWalkSampler(
            num_walks=num_walks,
            walk_length=walk_length,
            p=p,
            q=q,
        )

        self.sentences = None
        self.model = None
        self.verbose = verbose

        self.w2vparams = {
            "sg": 1,
            "min_count": 0,
            "epochs": 1,
            "workers": 4,
        }

    def fit(self, net):
        """Estimating the parameters for embedding.

        Parameters
        ---------
        net : nx.Graph object
            Network to be embedded. The graph type can be anything if
            the graph type is supported for the node samplers.

        Return
        ------
        self : Node2Vec
        """
        A = utils.to_adjacency_matrix(net)
        self.sampler.sampling(A)
        return self

    def update_embedding(self, dim):
        # Update the dimension and train the model
        # Sample the sequence of nodes using a random walk
        self.w2vparams["window"] = self.window_length

        self.w2vparams["vector_size"] = dim
        self.model = gensim.models.Word2Vec(
            sentences=self.sampler.walks, **self.w2vparams
        )

        num_nodes = len(self.model.wv.index_to_key)
        self.in_vec = np.zeros((num_nodes, dim))
        self.out_vec = np.zeros((num_nodes, dim))
        for i in range(num_nodes):
            if i not in self.model.wv:
                continue
            self.in_vec[i, :] = self.model.wv[i]
            self.out_vec[i, :] = self.model.syn1neg[self.model.wv.key_to_index[i]]


class DeepWalk(Node2Vec):
    def __init__(self, **params):
        Node2Vec.__init__(self, **params)
        self.w2vparams = {
            "sg": 1,
            "hs": 1,
            "min_count": 0,
            "workers": 4,
        }


class LaplacianEigenMap(NodeEmbeddings):
    def __init__(self, p=100, q=40, reconstruction_vector=False):
        self.in_vec = None
        self.L = None
        self.deg = None
        self.p = p
        self.q = q
        self.reconstruction_vector = reconstruction_vector

    def fit(self, G):
        A = utils.to_adjacency_matrix(G)

        # Compute the (inverse) normalized laplacian matrix
        deg = np.array(A.sum(axis=1)).reshape(-1)
        Dsqrt = sparse.diags(1 / np.maximum(np.sqrt(deg), 1e-12), format="csr")
        L = Dsqrt @ A @ Dsqrt

        self.L = L
        self.deg = deg
        return self

    def transform(self, dim, return_out_vector=False):
        if self.in_vec is None:
            self.update_embedding(dim)
        elif self.in_vec.shape[1] != dim:
            self.update_embedding(dim)
        return self.in_vec

    def update_embedding(self, dim):
        if self.reconstruction_vector:
            #            u, s, v = rsvd.rSVD(
            #                self.L, dim, p=self.p, q=self.q
            #            )  # add one for the trivial solution
            #            sign = np.sign(np.diag(v @ u))
            #            s = s * sign
            #            order = np.argsort(s)[::-1]
            #            u = u[:, order] @ np.diag(np.sqrt(np.maximum(0, s[order])))
            s, u = sparse.linalg.eigs(self.L, k=dim, which="LR")
            s, u = np.real(s), np.real(u)
            order = np.argsort(s)[::-1]
            self.in_vec = u[:, order]
            self.out_vec = u[:, order]
        else:
            s, u = sparse.linalg.eigs(self.L, k=dim + 1, which="LR")
            s, u = np.real(s), np.real(u)
            order = np.argsort(-s)[1:]
            s, u = s[order], u[:, order]
            #            u, s, v = rsvd.rSVD(
            #                self.L, dim + 1, p=self.p, q=self.q
            #            )  # add one for the trivial solution
            #            sign = np.sign(np.diag(v @ u))
            #            s = s * sign
            #            order = np.argsort(s)[::-1][1:]
            #            u = u[:, order]
            Dsqrt = sparse.diags(1 / np.maximum(np.sqrt(self.deg), 1e-12), format="csr")
            self.in_vec = Dsqrt @ u @ sparse.diags(np.sqrt(np.abs(s)))
            self.out_vec = u


class AdjacencySpectralEmbedding(NodeEmbeddings):
    def __init__(
        self,
        verbose=False,
    ):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)
        self.A = A
        return self

    def update_embedding(self, dim):
        svd = TruncatedSVD(n_components=dim, n_iter=7, random_state=42)
        u = svd.fit_transform(self.A)
        s = svd.singular_values_
        # u, s, v = rsvd.rSVD(self.A, dim=dim)
        self.in_vec = u @ sparse.diags(s)


class ModularitySpectralEmbedding(NodeEmbeddings):
    def __init__(self, verbose=False, reconstruction_vector=False, p=100, q=40):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.reconstruction_vector = reconstruction_vector
        self.p = p
        self.q = q

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)
        self.A = A
        self.deg = np.array(A.sum(axis=1)).reshape(-1)
        return self

    def update_embedding(self, dim):
        s, u = sparse.linalg.eigs(self.A, k=dim + 1, which="LR")
        s, u = np.real(s), np.real(u)
        s = s[1:]
        u = u[:, 1:]

        if self.reconstruction_vector:
            is_positive = s > 0
            u[:, ~is_positive] = 0
            s[~is_positive] = 0
            self.in_vec = u @ sparse.diags(np.sqrt(s))
        else:
            self.in_vec = u @ sparse.diags(np.sqrt(np.abs(s)))
        self.out_vec = u


class HighOrderModularitySpectralEmbedding(NodeEmbeddings):
    def __init__(
        self,
        verbose=False,
        window_length=10,
    ):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.window_length = window_length

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)
        self.A = A
        self.deg = np.array(A.sum(axis=1)).reshape(-1)
        return self

    def update_embedding(self, dim):
        stationary_prob = self.deg / np.sum(self.deg)

        P = utils.to_trans_mat(self.A)
        Q = []
        for t in range(self.window_length):
            Q.append(
                [sparse.diags(stationary_prob / self.window_length) @ P]
                + [P for _ in range(t)]
            )
        Q.append([-stationary_prob.reshape((-1, 1)), stationary_prob.reshape((1, -1))])
        u, s, v = rsvd.rSVD(Q, dim=dim)
        self.in_vec = u @ sparse.diags(s)
        self.out_vec = None


class LinearizedNode2Vec(NodeEmbeddings):
    def __init__(self, verbose=False, window_length=10, p=100, q=40):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.window_length = window_length
        self.p = p
        self.q = q

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)
        self.A = A
        self.deg = np.array(A.sum(axis=1)).reshape(-1)
        return self

    def update_embedding(self, dim):
        # Calculate the normalized transition matrix
        Dinvsqrt = sparse.diags(1 / np.sqrt(np.maximum(1, self.deg)))
        Psym = Dinvsqrt @ self.A @ Dinvsqrt

        # svd = TruncatedSVD(n_components=dim + 1, n_iter=7, random_state=42)
        # u = svd.fit_transform(Psym)
        # s = svd.singular_values_
        s, u = sparse.linalg.eigs(Psym, k=dim + 1, which="LR")
        s, u = np.real(s), np.real(u)
        order = np.argsort(-s)
        s, u = s[order], u[:, order]

        # u, s, v = rsvd.rSVD(Psym, dim=dim + 1, p=self.p, q=self.q)
        # sign = np.sign(np.diag(v @ u))

        s = np.abs(s)
        mask = s < np.max(s)
        u = u[:, mask]
        s = s[mask]

        if self.window_length > 1:
            s = (s * (1 - s**self.window_length)) / (self.window_length * (1 - s))

        self.in_vec = u @ sparse.diags(np.sqrt(np.abs(s)))
        self.out_vec = None


class NonBacktrackingSpectralEmbedding(NodeEmbeddings):
    def __init__(self, verbose=False, auto_dim=False):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.auto_dim = auto_dim
        self.C = 10

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)
        self.deg = np.array(A.sum(axis=1)).reshape(-1)
        self.A = A
        return self

    def update_embedding(self, dim):
        N = self.A.shape[0]
        Z = sparse.csr_matrix((N, N))
        I = sparse.identity(N, format="csr")
        D = sparse.diags(self.deg)
        B = sparse.bmat([[Z, D - I], [-I, self.A]], format="csr")

        if self.auto_dim is False:
            s, v = sparse.linalg.eigs(B, k=dim, tol=1e-4)
            s, v = np.real(s), np.real(v)
            order = np.argsort(-np.abs(s))
            s, v = s[order], v[:, order]
            v = v[N:, :]

            # Normalize the eigenvectors because we cut half the vec
            # and omit the imaginary part.
            c = np.array(np.linalg.norm(v, axis=0)).reshape(-1)
            v = v @ np.diag(1 / c)

            # Weight the dimension by the eigenvalues
            v = v @ np.diag(np.sqrt(np.abs(s)))
        else:
            dim = int(self.C * np.sqrt(N))
            dim = np.minimum(dim, N - 1)

            s, v = sparse.linalg.eigs(B, k=dim + 1, tol=1e-4)

            c = int(self.A.sum() / N)
            s, v = s[np.abs(s) > c], v[:, np.abs(s) > c]

            order = np.argsort(s)
            s, v = s[order], v[:, order]
            s, v = s[1:], v[:, 1:]
            v = v[N:, :]
            c = np.array(np.linalg.norm(v, axis=0)).reshape(-1)
            v = v @ np.diag(1 / c)

        self.in_vec = v


class Node2VecMatrixFactorization(NodeEmbeddings):
    def __init__(self, verbose=False, window_length=10, num_blocks=500):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.window_length = window_length
        self.num_blocks = num_blocks

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)

        self.A = A
        self.deg = np.array(A.sum(axis=1)).reshape(-1)
        return self

    def update_embedding(self, dim):
        P = utils.to_trans_mat(self.A)
        Ppow = utils.matrix_sum_power(P, self.window_length) / self.window_length
        stationary_prob = self.deg / np.sum(self.deg)
        R = np.log(Ppow @ np.diag(1 / stationary_prob))

        # u, s, v = rsvd.rSVD(R, dim=dim)
        svd = TruncatedSVD(n_components=dim + 1, n_iter=7, random_state=42)
        u = svd.fit_transform(R)
        s = svd.singular_values_
        self.in_vec = u @ sparse.diags(np.sqrt(s))
        self.out_vec = None


class NonBacktrackingNode2Vec(Node2Vec):
    """A python class for the node2vec.

    Parameters
    ----------
    num_walks : int (optional, default 10)
        Number of walks per node
    walk_length : int (optional, default 40)
        Length of walks
    window_length : int (optional, default 10)
        Restart probability of a random walker.
    p : node2vec parameter (TODO: Write doc)
    q : node2vec parameter (TODO: Write doc)
    """

    def __init__(self, num_walks=10, walk_length=80, window_length=10, **params):
        Node2Vec.__init__(
            self,
            num_walks=num_walks,
            walk_length=walk_length,
            window_length=window_length,
            **params
        )
        self.sampler = samplers.NonBacktrackingWalkSampler(
            num_walks=num_walks, walk_length=walk_length
        )


class NonBacktrackingDeepWalk(DeepWalk):
    """A python class for the node2vec.

    Parameters
    ----------
    num_walks : int (optional, default 10)
        Number of walks per node
    walk_length : int (optional, default 40)
        Length of walks
    window_length : int (optional, default 10)
        Restart probability of a random walker.
    p : node2vec parameter (TODO: Write doc)
    q : node2vec parameter (TODO: Write doc)
    """

    def __init__(self, num_walks=10, walk_length=80, window_length=10, **params):
        DeepWalk.__init__(
            self,
            num_walks=num_walks,
            walk_length=walk_length,
            window_length=window_length,
            **params
        )
        self.sampler = samplers.NonBacktrackingWalkSampler(
            num_walks=num_walks, walk_length=walk_length
        )


class DegreeEmbedding:
    def __init__(self, **params):
        return

    def fit(self, net):
        self.degree = np.array(net.sum(axis=0)).reshape(-1)

    def transform(self, dim):
        emb = np.zeros((len(self.degree), dim))
        emb[:, 0] = self.degree
        return emb


class GAT(torch.nn.Module):
    """A python class for the GAT.

    Parameters
    ----------
    dim_in: dimension of in vector
    dim_out: dimension of out vector
    dim_h : dimension of hidden layer

    """

    def __init__(self, dim_in, dim_h, dim_out):
        super(GAT, self).__init__()
        self.conv1 = GATv2Conv(dim_in, dim_h)
        self.conv2 = GATv2Conv(dim_h, dim_out)

    def forward(self, x, positive_edge_index):
        h = self.conv1(x, positive_edge_index)
        h = h.relu()
        h = self.conv2(h, positive_edge_index)
        return h

    def decode(self, z, pos_edge_index, neg_edge_index):  # only pos and neg edges
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)  # dot product
        return logits


class GAT_model(torch.nn.Module):
    def __init__(self, feature_dim, dim_h, device, epochs):
        self.feature_dim = feature_dim
        self.dim_h = dim_h
        self.device = device
        self.epochs = epochs

    def fit(self, net):
        self.network = net
        model = LaplacianEigenMap()
        model.fit(self.network)
        self.features = model.transform(dim=self.feature_dim)

    def transform(self, dim):
        model_GAT, data = GAT(self.feature_dim, self.dim_h, dim).to(
            self.device
        ), torch.from_numpy(self.features).to(dtype=torch.float, device=self.device)
        model_trained = train.train(
            model_GAT, data, self.network, self.device, self.epochs
        )

        network_c = self.network.tocoo()

        edge_list_gat = torch.from_numpy(np.array([network_c.row, network_c.col])).to(
            self.device
        )

        embeddings = model_trained(data, edge_list_gat)

        return embeddings


class GCN(torch.nn.Module):
    """A python class for the GCN.

    Parameters
    ----------
    dim_in: dimension of in vector
    dim_out: dimension of out vector
    dim_h : dimension of hidden layer

    """

    def __init__(self, dim_in, dim_h, dim_out):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dim_in, dim_h)
        self.conv2 = GCNConv(dim_h, dim_out)

    def forward(self, x, positive_edge_index):
        h = self.conv1(x, positive_edge_index)
        h = h.relu()
        h = self.conv2(h, positive_edge_index)
        return h

    def decode(self, z, pos_edge_index, neg_edge_index):  # only pos and neg edges
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)  # dot product
        return logits


class GCN_model(torch.nn.Module):
    def __init__(self, feature_dim, dim_h, device, epochs):
        self.feature_dim = feature_dim
        self.dim_h = dim_h
        self.device = device
        self.epochs = epochs

    def fit(self, net):
        self.network = net
        model = LaplacianEigenMap()
        model.fit(self.network)
        self.features = model.transform(dim=self.feature_dim)

    def transform(self, dim):
        model_GCN, data = GCN(self.feature_dim, self.dim_h, dim).to(
            self.device
        ), torch.from_numpy(self.features).to(dtype=torch.float, device=self.device)
        model_trained = train.train(
            model_GCN, data, self.network, self.device, self.epochs
        )

        network_c = self.network.tocoo()

        edge_list_gcn = torch.from_numpy(np.array([network_c.row, network_c.col])).to(
            self.device
        )

        embeddings = model_trained(data, edge_list_gcn)

        return embeddings
