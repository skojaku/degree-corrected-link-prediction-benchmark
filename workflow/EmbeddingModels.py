# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 15:08:01
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-03-31 17:44:39

import embcom
import torch
import numpy as np

embedding_models = {}
embedding_model = lambda f: embedding_models.setdefault(f.__name__, f)


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
def GCN(network, dim, feature_dim=10, device="cpu", dim_h=128, epochs=2000):
    """
    Parameters
    ----------
    network: adjacency matrix
    feature_dim: dimension of features
    dim: dimension of embedding vectors
    dim_h : dimension of hidden layer
    device : device
    epochs : the number of epoch in training
    """
    model = embcom.embeddings.GCN_model(
        feature_dim=feature_dim, dim_h=dim_h, device=device, epochs=epochs
    )
    model.fit(network)
    return model.transform(dim=dim)


@embedding_model
def GAT(network, dim, feature_dim=10, device="cpu", dim_h=128, epochs=2000):
    """
    Parameters
    ----------
    network: adjacency matrix
    feature_dim: dimension of features
    dim: dimension of embedding vectors
    dim_h : dimension of hidden layer
    device : device
    epochs : the number of epoch in training
    """
    model = embcom.embeddings.GAT_model(
        feature_dim=feature_dim, dim_h=dim_h, device=device, epochs=epochs
    )
    model.fit(network)
    return model.transform(dim=dim)

# @embedding_model
# def nonbacktracking(network, dim):
#    model = embcom.embeddings.NonBacktrackingSpectralEmbedding()
#    model.fit(network)
#    return model.transform(dim=dim)

@embedding_model
def graphsage(network, num_walks=1, walk_length=5, dim=10):
    model = embcom.embeddings.graphSAGE(num_walks=num_walks, walk_length=walk_length, dim=dim)
    model.fit(network)
    model.train_GraphSAGE()
    return model.get_embeddings()

