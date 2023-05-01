# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-04-22 20:48:19
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-04-22 21:18:41
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
import embcom
import torch
import numpy as np


def toUndirected(net):
    net = net + net.T
    net.data = net.data * 0 + 1
    net.eliminate_zeros()
    net = sparse.csr_matrix.asfptype(net)
    return net


def edge2network(src, trg, n_nodes=None, val=None):
    if val is None:
        val = np.ones_like(src)
    if n_nodes is None:
        n_nodes = np.max([np.max(src), np.max(trg)]) + 1
    return toUndirected(sparse.csr_matrix((val, (src, trg)), shape=(n_nodes, n_nodes)))


#
# Embedding models
#
embedding_models = {}
embedding_model = lambda f: embedding_models.setdefault(f.__name__, f)


def calc_prob_i_j(emb, src, trg, net, model_name):
    score = np.sum(emb[src, :] * emb[trg, :], axis=1).reshape(-1)

    # We want to calculate the probability P(i,j) of
    # random walks moving from i to j, instead of the dot similarity.
    # The P(i,j) is given by
    #    P(i,j) \propto \exp(u[i]^\top u[j] + \ln p0[i] + \ln p0[j])
    # where p0 is proportional to the degree. In residual2vec paper,
    # we found that P(i,j) is more predictable of missing edges than
    # the dot similarity u[i]^\top u[j].
    if model_name in ["deepwalk", "node2vec", "line", "graphsage"]:
        deg = np.array(net.sum(axis=1)).reshape(-1)
        deg = np.maximum(deg, 1)
        deg = deg / np.sum(deg)
        log_deg = np.log(deg)
        score += log_deg[src] + log_deg[trg]
    return score


@embedding_model
def line(network, dim, num_walks=40):
    model = embcom.embeddings.Node2Vec(window_length=1, num_walks=num_walks)
    model.fit(network)
    return model.transform(dim=dim)


@embedding_model
def node2vec(network, dim, window_length=10, num_walks=40):
    model = embcom.embeddings.Node2Vec(window_length=window_length, num_walks=num_walks)
    model.fit(network)
    return model.transform(dim=dim)


@embedding_model
def deepwalk(network, dim, window_length=10, num_walks=40):
    model = embcom.embeddings.DeepWalk(window_length=window_length, num_walks=num_walks)
    model.fit(network)
    return model.transform(dim=dim)


@embedding_model
def leigenmap(network, dim):
    model = embcom.embeddings.LaplacianEigenMap()
    model.fit(network)
    return model.transform(dim=dim)


@embedding_model
def modspec(network, dim):
    model = embcom.embeddings.ModularitySpectralEmbedding()
    model.fit(network)
    return model.transform(dim=dim)


@embedding_model
def GCN(network, dim, feature_dim=64, device="cuda:0", dim_h=128):
    """
    Parameters
    ----------
    network: adjacency matrix
    feature_dim: dimension of features
    dim: dimension of embedding vectors
    dim_h : dimension of hidden layer
    device : device

    """

    model = embcom.embeddings.LaplacianEigenMap()
    model.fit(network)
    features = model.transform(dim=feature_dim)

    model_GCN, data = embcom.embeddings.GCN(feature_dim, dim_h, dim).to(
        device
    ), torch.from_numpy(features).to(dtype=torch.float, device=device)
    model_trained = embcom.train(model_GCN, data, network, device)

    network_c = network.tocoo()

    edge_list_gcn = torch.from_numpy(np.array([network_c.row, network_c.col])).to(
        device
    )

    embeddings = model_trained(data, edge_list_gcn)

    return embeddings.detach().cpu().numpy()


@embedding_model
def nonbacktracking(network, dim):
    model = embcom.embeddings.NonBacktrackingSpectralEmbedding()
    model.fit(network)
    return model.transform(dim=dim)


# @embedding_model
# def graphsage(network, dim, num_walks=1, walk_length=5):
#    model = embcom.embeddings.graphSAGE(
#        num_walks=num_walks, walk_length=walk_length, emb_dim=dim
#    )
#    model.fit(network)
#    model.train_GraphSAGE()
#    return model.get_embeddings()


@embedding_model
def fastrp(network, dim, window_length=5, inner_dim=2048):
    model = embcom.embeddings.FastRP(window_size=5)
    model.fit(network)
    emb = model.transform(dim=inner_dim)
    return PCA(n_components=dim).fit_transform(emb)


#
# Network topology
#
topology_models = {}
topology_model = lambda f: topology_models.setdefault(f.__name__, f)


@topology_model
def preferentialAttachment(network, src, trg):
    deg = np.array(network.sum(axis=1)).reshape(-1)
    return deg[src] * deg[trg]


@topology_model
def commonNeighbors(network, src, trg):
    return np.array((network[src, :].multiply(network[trg, :])).sum(axis=1)).reshape(-1)


@topology_model
def jaccardIndex(network, src, trg):
    deg = np.array(network.sum(axis=1)).reshape(-1)
    score = np.array((network[src, :].multiply(network[trg, :])).sum(axis=1)).reshape(
        -1
    )
    return score / np.maximum(deg[src] + deg[trg] - score, 1)


@topology_model
def resourceAllocation(network, src, trg):
    deg = np.array(network.sum(axis=1)).reshape(-1)
    deg_inv = 1 / np.maximum(deg, 1)
    deg_inv[deg == 0] = 0
    return np.array(
        ((network[src, :] @ sparse.diags(deg_inv)).multiply(network[trg, :])).sum(
            axis=1
        )
    ).reshape(-1)


@topology_model
def adamicAdar(network, src, trg):
    deg = np.array(network.sum(axis=1)).reshape(-1)
    log_deg_inv = 1 / np.maximum(np.log(np.maximum(deg, 1)), 1)
    log_deg_inv[deg == 0] = 0
    return np.array(
        ((network[src, :] @ sparse.diags(log_deg_inv)).multiply(network[trg, :])).sum(
            axis=1
        )
    ).reshape(-1)


@topology_model
def localRandomWalk(network, src, trg):
    deg = np.array(network.sum(axis=1)).reshape(-1)
    deg_inv = 1 / np.maximum(deg, 1)
    P = sparse.diags(deg_inv) @ network
    PP = P @ P
    PPP = PP @ P
    S = P + PP + PPP
    S = sparse.diags(deg / np.sum(deg)) @ S
    return np.array(S[(src, trg)]).reshape(-1)


@topology_model
def localPathIndex(network, src, trg, epsilon=1e-3):
    A = network
    AA = A @ A
    AAA = AA @ A
    S = AA + epsilon * AAA
    return np.array(S[(src, trg)]).reshape(-1)
