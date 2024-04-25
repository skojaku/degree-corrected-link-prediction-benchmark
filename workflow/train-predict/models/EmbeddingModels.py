# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 15:08:01
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-07-29 06:06:37
# %%
from sklearn.decomposition import PCA
import graph_embedding
import torch
import numpy as np
import torch_geometric
import torch
from .ModelTemplate import LinkPredictor


class EmbeddingLinkPredictor(LinkPredictor):
    def __init__(self, model, **params):
        super().__init__()
        self.model = model
        self.params = params
        self.embedding_models = embedding_models

    def train(self, network, **params):
        emb_func = embedding_models[self.model]
        emb = emb_func(network=network, **self.params)
        self.emb = torch.nn.Parameter(torch.FloatTensor(emb), requires_grad=False)

    def predict(self, network, src, trg, **params):
        return torch.sum(self.emb[src, :] * self.emb[trg, :], axis=1).reshape(-1)

    def load(self, filename):
        d = torch.load(filename)
        self.model = d["model"]
        self.emb = d["emb"]


#
# Embedding models
#
embedding_models = {}
embedding_model = lambda f: embedding_models.setdefault(f.__name__, f)


@embedding_model
def line(network, dim, num_walks=40, **params):
    model = graph_embedding.embeddings.Node2Vec(window_length=1, num_walks=num_walks)
    model.fit(network)
    return model.transform(dim=dim)


@embedding_model
def node2vec(network, dim, window_length=10, num_walks=40, **params):
    model = graph_embedding.embeddings.Node2Vec(
        window_length=window_length, num_walks=num_walks
    )
    model.fit(network)
    return model.transform(dim=dim)


@embedding_model
def deepwalk(network, dim, window_length=10, num_walks=40, **params):
    model = graph_embedding.embeddings.DeepWalk(
        window_length=window_length, num_walks=num_walks
    )
    model.fit(network)
    return model.transform(dim=dim)


@embedding_model
def leigenmap(network, dim, **params):
    model = graph_embedding.embeddings.LaplacianEigenMap()
    model.fit(network)
    return model.transform(dim=dim)


@embedding_model
def modspec(network, dim, **params):
    model = graph_embedding.embeddings.ModularitySpectralEmbedding()
    model.fit(network)
    return model.transform(dim=dim)


@embedding_model
def nonbacktracking(network, dim, **params):
    model = graph_embedding.embeddings.NonBacktrackingSpectralEmbedding()
    model.fit(network)
    return model.transform(dim=dim)


@embedding_model
def fastrp(network, dim, window_length=5, inner_dim=2048, **params):
    model = graph_embedding.embeddings.FastRP(window_size=window_length)
    model.fit(network)
    emb = model.transform(dim=inner_dim)
    return PCA(n_components=dim).fit_transform(emb)


@embedding_model
def SGTLaplacianExp(network, dim, **params):
    model = graph_embedding.embeddings.SpectralGraphTransformation(
        kernel_func="exp", kernel_matrix="laplacian"
    )
    model.fit(network)
    emb = model.transform(dim=dim)
    return emb


@embedding_model
def SGTLaplacianNeumann(network, dim, **params):
    model = graph_embedding.embeddings.SpectralGraphTransformation(
        kernel_func="neu", kernel_matrix="laplacian"
    )
    model.fit(network)
    emb = model.transform(dim=dim)
    return emb


@embedding_model
def SGTAdjacencyExp(network, dim, **params):
    model = graph_embedding.embeddings.SpectralGraphTransformation(
        kernel_func="exp", kernel_matrix="A"
    )
    model.fit(network)
    emb = model.transform(dim=dim)
    return emb


@embedding_model
def SGTAdjacencyNeumann(network, dim, **params):
    model = graph_embedding.embeddings.SpectralGraphTransformation(
        kernel_func="neu", kernel_matrix="A"
    )
    model.fit(network)
    emb = model.transform(dim=dim)
    return emb


@embedding_model
def SGTNormAdjacencyExp(network, dim, **params):
    model = graph_embedding.embeddings.SpectralGraphTransformation(
        kernel_func="exp", kernel_matrix="normalized_A"
    )
    model.fit(network)
    emb = model.transform(dim=dim)
    return emb


@embedding_model
def SGTNormAdjacencyNeumann(network, dim, **params):
    model = graph_embedding.embeddings.SpectralGraphTransformation(
        kernel_func="neu", kernel_matrix="normalized_A"
    )
    model.fit(network)
    emb = model.transform(dim=dim)
    return emb


@embedding_model
def dcSBM(network, dim, **params):
    model = graph_embedding.embeddings.SBMEmbedding()
    model.fit(network)
    emb = model.transform(dim=dim)
    return emb


#
# Generic graph neural networks
#
def gnn_embedding(
    model, network, device=None, epochs=50, negative_edge_sampler=None, **params
):
    if device is None:
        device = graph_embedding.gnns.get_gpu_id()

    model, emb = graph_embedding.gnns.train(
        model=model,
        feature_vec=None,
        net=network,
        negative_edge_sampler=negative_edge_sampler,
        device=device,
        epochs=epochs,
    )
    return emb


@embedding_model
def GCN(
    network,
    dim,
    feature_dim=64,
    num_layers=2,
    device=None,
    dim_h=64,
    epochs=50,
    **params
):
    return gnn_embedding(
        model=torch_geometric.nn.models.GCN(
            in_channels=feature_dim,
            hidden_channels=dim_h,
            num_layers=num_layers,
            out_channels=dim,
        ),
        network=network,
        negative_edge_sampler=graph_embedding.gnns.NegativeEdgeSampler["uniform"],
        epochs=epochs,
    )


@embedding_model
def GIN(
    network,
    dim,
    feature_dim=64,
    device=None,
    dim_h=64,
    num_layers=2,
    epochs=50,
    **params
):
    return gnn_embedding(
        model=torch_geometric.nn.models.GIN(
            in_channels=feature_dim,
            hidden_channels=dim_h,
            num_layers=num_layers,
            out_channels=dim,
        ),
        network=network,
        negative_edge_sampler=graph_embedding.gnns.NegativeEdgeSampler["uniform"],
        device=device,
        epochs=epochs,
    )


@embedding_model
def PNA(
    network,
    dim,
    feature_dim=64,
    device=None,
    dim_h=64,
    num_layers=2,
    epochs=50,
    **params
):
    return gnn_embedding(
        model=torch_geometric.nn.models.PNA(
            in_channels=feature_dim,
            hidden_channels=dim_h,
            num_layers=num_layers,
            out_channels=dim,
            aggregators=["sum", "mean", "min", "max", "max", "var", "std"],
            scalers=[
                "identity",
                "amplification",
                "attenuation",
                "linear",
                "inverse_linear",
            ],
            deg=torch.FloatTensor(
                np.bincount(np.array(network.sum(axis=0)).reshape(-1))
            ),
        ),
        network=network,
        negative_edge_sampler=graph_embedding.gnns.NegativeEdgeSampler["uniform"],
        device=device,
        epochs=epochs,
    )


@embedding_model
def EdgeCNN(
    network,
    dim,
    feature_dim=64,
    device=None,
    dim_h=64,
    num_layers=2,
    epochs=50,
    **params
):
    return gnn_embedding(
        model=torch_geometric.nn.models.EdgeCNN(
            in_channels=feature_dim,
            hidden_channels=dim_h,
            num_layers=num_layers,
            out_channels=dim,
        ),
        network=network,
        negative_edge_sampler=graph_embedding.gnns.NegativeEdgeSampler["uniform"],
        device=device,
        epochs=epochs,
    )


@embedding_model
def GraphSAGE(
    network,
    dim,
    feature_dim=64,
    device=None,
    dim_h=64,
    num_layers=2,
    epochs=50,
    **params
):
    return gnn_embedding(
        model=torch_geometric.nn.models.GraphSAGE(
            in_channels=feature_dim,
            hidden_channels=dim_h,
            num_layers=num_layers,
            out_channels=dim,
        ),
        network=network,
        negative_edge_sampler=graph_embedding.gnns.NegativeEdgeSampler["uniform"],
        device=device,
        epochs=epochs,
    )


@embedding_model
def GAT(
    network,
    dim,
    feature_dim=64,
    num_layers=2,
    device=None,
    dim_h=64,
    epochs=50,
    **params
):
    return gnn_embedding(
        model=torch_geometric.nn.models.GAT(
            in_channels=feature_dim,
            hidden_channels=dim_h,
            num_layers=num_layers,
            out_channels=dim,
        ),
        network=network,
        negative_edge_sampler=graph_embedding.gnns.NegativeEdgeSampler["uniform"],
        epochs=epochs,
    )
