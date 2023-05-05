# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 15:08:01
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-04 22:36:58

from sklearn.decomposition import PCA
import embcom
import torch
import numpy as np

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
    #    if model_name in ["deepwalk", "node2vec", "line", "graphsage"]:
    #        deg = np.array(net.sum(axis=1)).reshape(-1)
    #        deg = np.maximum(deg, 1)
    #        deg = deg / np.sum(deg)
    #        log_deg = np.log(deg)
    #        score += log_deg[src] + log_deg[trg]
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


# @embedding_model
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


#
#
@embedding_model
def nonbacktracking(network, dim):
    model = embcom.embeddings.NonBacktrackingSpectralEmbedding()
    model.fit(network)
    return model.transform(dim=dim)


# @embedding_model
# def graphsage(network, dim, num_walks=1, walk_length=5):
#   model = embcom.embeddings.graphSAGE(
#       num_walks=num_walks, walk_length=walk_length, emb_dim=dim
#   )
#   model.fit(network)
#   model.train_GraphSAGE()
#   return model.get_embeddings()


@embedding_model
def fastrp(network, dim, window_length=5, inner_dim=2048):
    model = embcom.embeddings.FastRP(window_size=5)
    model.fit(network)
    emb = model.transform(dim=inner_dim)
    return PCA(n_components=dim).fit_transform(emb)


@embedding_model
def SGTLaplacianExp(network, dim):
    model = embcom.embeddings.SpectralGraphTransformation(
        kernel_func="exp", kernel_matrix="laplacian"
    )
    model.fit(network)
    emb = model.transform(dim=dim)
    return emb


@embedding_model
def SGTLaplacianNeumann(network, dim):
    model = embcom.embeddings.SpectralGraphTransformation(
        kernel_func="neu", kernel_matrix="laplacian"
    )
    model.fit(network)
    emb = model.transform(dim=dim)
    return emb


@embedding_model
def SGTAdjacencyExp(network, dim):
    model = embcom.embeddings.SpectralGraphTransformation(
        kernel_func="exp", kernel_matrix="A"
    )
    model.fit(network)
    emb = model.transform(dim=dim)
    return emb


@embedding_model
def SGTAdjacencyNeumann(network, dim):
    model = embcom.embeddings.SpectralGraphTransformation(
        kernel_func="neu", kernel_matrix="A"
    )
    model.fit(network)
    emb = model.transform(dim=dim)
    return emb


@embedding_model
def SGTNormAdjacencyExp(network, dim):
    model = embcom.embeddings.SpectralGraphTransformation(
        kernel_func="exp", kernel_matrix="normalized_A"
    )
    model.fit(network)
    emb = model.transform(dim=dim)
    return emb


@embedding_model
def SGTNormAdjacencyNeumann(network, dim):
    model = embcom.embeddings.SpectralGraphTransformation(
        kernel_func="neu", kernel_matrix="normalized_A"
    )
    model.fit(network)
    emb = model.transform(dim=dim)
    return emb
