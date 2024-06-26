# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 15:08:01
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-15 16:09:32

from sklearn.decomposition import PCA
import embcom
import torch
import numpy as np
import torch_geometric

embedding_models = {}
embedding_model = lambda f: embedding_models.setdefault(f.__name__, f)

degree_corrected_gnn_models = [
    "node2vec",
    "line",
    "dcGCN",
    "dcGraphSAGE",
    "dcGAT",
    "dcGIN",
    "dcPNA",
    "dcEdgeCNN",
    # "dcGraphUNet",
]


def calc_prob_i_j(emb, src, trg, net, model_name):
    score = np.sum(emb[src, :] * emb[trg, :], axis=1).reshape(-1)
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
def nonbacktracking(network, dim):
    model = embcom.embeddings.NonBacktrackingSpectralEmbedding()
    model.fit(network)
    return model.transform(dim=dim)


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


#
# Graph neural networks
#
# def gnn_embedding(model, network, feature_vec_dim=64, device=None, epochs=2000, negative_edge_sampler=None, batch_size=2500):
#    if device is None:
#        device = embcom.gnns.get_gpu_id()
#
#    model, emb = embcom.gnns.train(
#        model=model,
#        feature_vec=None,
#        net=network,
#        negative_edge_sampler=negative_edge_sampler,
#        device=device,
#        epochs=epochs,
#        batch_size=batch_size,
#        feature_vec_dim=feature_vec_dim
#    )
#    return emb


def gnn_embedding(
    model,
    network,
    feature_vec_dim,
    device=None,
    epochs=1000,
    negative_edge_sampler=None,
    batch_size=3500 * 3,
):
    if device is None:
        device = embcom.gnns.get_gpu_id()
    model_ext = embcom.gnns.GNNwithEmbLayer(
        gnn=model,
        dim_emb=feature_vec_dim,
        n_nodes=network.shape[0],
    )
    model, emb = embcom.gnns.train(
        model=model_ext,
        feature_vec=None,  # network.toarray(),
        feature_vec_dim=feature_vec_dim,
        net=network,
        negative_edge_sampler=negative_edge_sampler,
        device=device,
        epochs=epochs,
        # batch_size=batch_size,
        lr=1e-3,
    )
    return model, emb


@embedding_model
def GCN(network, dim, feature_dim=64, device=None, dim_h=128):
    return gnn_embedding(
        model=torch_geometric.nn.models.GCN(
            in_channels=feature_dim,
            hidden_channels=dim_h,
            num_layers=2,
            out_channels=dim,
            act=torch.nn.LeakyReLU(),
            dropout=0.5,
        ),
        feature_vec_dim=feature_dim,
        network=network,
    )


@embedding_model
def GraphSAGE(network, dim, feature_dim=64, device=None, dim_h=128):
    return gnn_embedding(
        model=torch_geometric.nn.models.GraphSAGE(
            in_channels=feature_dim,
            hidden_channels=dim_h,
            num_layers=2,
            act=torch.nn.LeakyReLU(),
            out_channels=dim,
            dropout=0.5,
        ),
        network=network,
    )


@embedding_model
def GAT(network, dim, feature_dim=64, device=None, dim_h=128):
    return gnn_embedding(
        model=torch_geometric.nn.models.GAT(
            in_channels=feature_dim,
            hidden_channels=dim_h,
            num_layers=2,
            out_channels=dim,
            act=torch.nn.LeakyReLU(),
            dropout=0.5,
        ),
        feature_vec_dim=feature_dim,
        network=network,
    )


@embedding_model
def GIN(network, dim, feature_dim=64, device=None, dim_h=128):
    return gnn_embedding(
        model=torch_geometric.nn.models.GIN(
            in_channels=feature_dim,
            hidden_channels=dim_h,
            num_layers=2,
            out_channels=dim,
            act=torch.nn.LeakyReLU(),
            dropout=0.5,
        ),
        network=network,
    )


@embedding_model
def PNA(network, dim, feature_dim=64, device=None, dim_h=128):
    return gnn_embedding(
        model=torch_geometric.nn.models.PNA(
            in_channels=feature_dim,
            hidden_channels=dim_h,
            num_layers=2,
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
                np.bincount(np.array(network.sum(axis=0)).reshape(-1).astype(int))
            ),
            act=torch.nn.LeakyReLU(),
            dropout=0.5,
        ),
        feature_vec_dim=feature_dim,
        network=network,
    )


@embedding_model
def EdgeCNN(network, dim, feature_dim=64, device=None, dim_h=128):
    return gnn_embedding(
        model=torch_geometric.nn.models.EdgeCNN(
            in_channels=feature_dim,
            hidden_channels=dim_h,
            num_layers=2,
            out_channels=dim,
            act=torch.nn.LeakyReLU(),
            dropout=0.5,
        ),
        feature_vec_dim=feature_dim,
        network=network,
    )


# @embedding_model
# def GraphUNet(network, dim, feature_dim=64, device=None, dim_h=128):
#    return gnn_embedding(
#        model=torch_geometric.nn.models.GraphUNet(
#            in_channels=feature_dim,
#            hidden_channels=dim_h,
#            out_channels=dim,
#            depth=1,
#        ),
#        batch_size = 2500,
#        network=network,
#    )


@embedding_model
def dcGCN(network, dim, feature_dim=64, device=None, dim_h=128):
    return gnn_embedding(
        model=torch_geometric.nn.models.GCN(
            in_channels=feature_dim,
            hidden_channels=dim_h,
            num_layers=2,
            out_channels=dim,
            act=torch.nn.LeakyReLU(),
            dropout=0.5,
        ),
        network=network,
        negative_edge_sampler=embcom.gnns.NegativeEdgeSampler["degreeBiased"],
    )


@embedding_model
def dcGraphSAGE(network, dim, feature_dim=64, device=None, dim_h=128):
    return gnn_embedding(
        model=torch_geometric.nn.models.GraphSAGE(
            in_channels=feature_dim,
            hidden_channels=dim_h,
            num_layers=2,
            out_channels=dim,
            act=torch.nn.LeakyReLU(),
            dropout=0.5,
        ),
        network=network,
        feature_vec_dim=feature_dim,
        negative_edge_sampler=embcom.gnns.NegativeEdgeSampler["degreeBiased"],
    )


@embedding_model
def dcGAT(network, dim, feature_dim=64, device=None, dim_h=128):
    return gnn_embedding(
        model=torch_geometric.nn.models.GAT(
            in_channels=feature_dim,
            hidden_channels=dim_h,
            num_layers=2,
            out_channels=dim,
            act=torch.nn.LeakyReLU(),
            dropout=0.5,
        ),
        network=network,
        feature_vec_dim=feature_dim,
        negative_edge_sampler=embcom.gnns.NegativeEdgeSampler["degreeBiased"],
    )


@embedding_model
def dcGIN(network, dim, feature_dim=64, device=None, dim_h=128):
    return gnn_embedding(
        model=torch_geometric.nn.models.GIN(
            in_channels=feature_dim,
            hidden_channels=dim_h,
            num_layers=2,
            out_channels=dim,
            act=torch.nn.LeakyReLU(),
            dropout=0.5,
        ),
        network=network,
        feature_vec_dim=feature_dim,
        negative_edge_sampler=embcom.gnns.NegativeEdgeSampler["degreeBiased"],
    )


@embedding_model
def dcPNA(network, dim, feature_dim=64, device=None, dim_h=128):
    return gnn_embedding(
        model=torch_geometric.nn.models.PNA(
            in_channels=feature_dim,
            hidden_channels=dim_h,
            num_layers=2,
            out_channels=dim,
            aggregators=["sum", "mean", "min", "max", "max", "var", "std"],
            scalers=[
                "identity",
                "amplification",
                "attenuation",
                "linear",
                "inverse_linear",
            ],
            act=torch.nn.LeakyReLU(),
            deg=torch.FloatTensor(
                np.bincount(np.array(network.sum(axis=0)).reshape(-1).astype(int))
            ),
            dropout=0.5,
        ),
        network=network,
        feature_vec_dim=feature_dim,
        negative_edge_sampler=embcom.gnns.NegativeEdgeSampler["degreeBiased"],
    )


@embedding_model
def dcEdgeCNN(network, dim, feature_dim=64, device=None, dim_h=128):
    return gnn_embedding(
        model=torch_geometric.nn.models.EdgeCNN(
            in_channels=feature_dim,
            hidden_channels=dim_h,
            num_layers=2,
            act=torch.nn.LeakyReLU(),
            out_channels=dim,
            dropout=0.5,
        ),
        network=network,
        feature_vec_dim=feature_dim,
        negative_edge_sampler=embcom.gnns.NegativeEdgeSampler["degreeBiased"],
    )


# @embedding_model
# def dcGraphUNet(network, dim, feature_dim=64, device=None, dim_h=128):
#    return gnn_embedding(
#        model=torch_geometric.nn.models.GraphUNet(
#            in_channels=feature_dim,
#            hidden_channels=dim_h,
#            out_channels=dim,
#            depth=1,
#        ),
#        batch_size = 2500,
#        network=network,
#        negative_edge_sampler=embcom.gnns.NegativeEdgeSampler["degreeBiased"],
#    )
