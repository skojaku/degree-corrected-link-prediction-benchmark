# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 15:08:01
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-13 11:53:34

from sklearn.decomposition import PCA
import embcom
import torch
import numpy as np

embedding_models = {}
embedding_model = lambda f: embedding_models.setdefault(f.__name__, f)

degree_corrected_gnn_models = ["node2vec", "line", "dcGCN", "dcGAT", "dcGraphSAGE"]


def calc_prob_i_j(emb, src, trg, net, model_name):
    score = np.sum(emb[src, :] * emb[trg, :], axis=1).reshape(-1)

    # We want to calculate the probability P(i,j) of
    # random walks moving from i to j, instead of the dot similarity.
    # The P(i,j) is given by
    #    P(i,j) \propto \exp(u[i]^\top u[j] + \ln p0[i] + \ln p0[j])
    # where p0 is proportional to the degree. In residual2vec paper,
    # we found that P(i,j) is more predictable of missing edges than
    # the dot similarity u[i]^\top u[j].
    if model_name in degree_corrected_gnn_models:
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
def GCN(network, dim, feature_dim=64, device=None, dim_h=128):
    if device is None:
        device = embcom.gnns.get_gpu_id()
    gnn = embcom.gnns.GCN(dim_in=feature_dim, dim_h=dim_h, dim_out=dim)
    gnn = embcom.gnns.train(
        model=gnn, feature_vec=None, net=network, device=device, epochs=500
    )
    return gnn.generate_embedding(feature_vec=None, net=network, device=device)


@embedding_model
def GraphSAGE(network, dim, feature_dim=64, device=None, dim_h=128):
    if device is None:
        device = embcom.gnns.get_gpu_id()
    gnn = embcom.gnns.GraphSAGE(dim_in=feature_dim, dim_h=dim_h, dim_out=dim)
    gnn = embcom.gnns.train(
        model=gnn, feature_vec=None, net=network, device=device, epochs=500
    )
    return gnn.generate_embedding(feature_vec=None, net=network, device=device)


@embedding_model
def GAT(network, dim, feature_dim=64, device=None, dim_h=128):
    if device is None:
        device = embcom.gnns.get_gpu_id()
    gnn = embcom.gnns.GAT(dim_in=feature_dim, dim_h=dim_h, dim_out=dim)
    gnn = embcom.gnns.train(
        model=gnn, feature_vec=None, net=network, device=device, epochs=500
    )
    return gnn.generate_embedding(feature_vec=None, net=network, device=device)


@embedding_model
def dcGCN(network, dim, feature_dim=64, device=None, dim_h=128):
    if device is None:
        device = embcom.gnns.get_gpu_id()
    gnn = embcom.gnns.GCN(dim_in=feature_dim, dim_h=dim_h, dim_out=dim)
    gnn = embcom.gnns.train(
        model=gnn,
        feature_vec=None,
        net=network,
        device=device,
        epochs=500,
        negative_edge_sampler=embcom.gnns.NegativeEdgeSampler["degreeBiased"],
    )
    return gnn.generate_embedding(feature_vec=None, net=network, device=device)


@embedding_model
def dcGraphSAGE(network, dim, feature_dim=64, device=None, dim_h=128):
    if device is None:
        device = embcom.gnns.get_gpu_id()
    gnn = embcom.gnns.GraphSAGE(dim_in=feature_dim, dim_h=dim_h, dim_out=dim)
    gnn = embcom.gnns.train(
        model=gnn,
        feature_vec=None,
        net=network,
        device=device,
        epochs=500,
        negative_edge_sampler=embcom.gnns.NegativeEdgeSampler["degreeBiased"],
    )
    return gnn.generate_embedding(feature_vec=None, net=network, device=device)


@embedding_model
def dcGAT(network, dim, feature_dim=64, device=None, dim_h=128):
    if device is None:
        device = embcom.gnns.get_gpu_id()
    gnn = embcom.gnns.GAT(dim_in=feature_dim, dim_h=dim_h, dim_out=dim)
    gnn = embcom.gnns.train(
        model=gnn,
        feature_vec=None,
        net=network,
        device=device,
        epochs=500,
        negative_edge_sampler=embcom.gnns.NegativeEdgeSampler["degreeBiased"],
    )
    return gnn.generate_embedding(feature_vec=None, net=network, device=device)


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


@embedding_model
def dcSBM(network, dim):
    model = embcom.embeddings.SBMEmbedding()
    model.fit(network)
    emb = model.transform(dim=dim)
    return emb
