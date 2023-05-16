# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-03-27 17:59:10
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-13 16:11:03
# %%
import embcom
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import sparse
import numpy as np


def load_airport_net():
    # Node attributes
    node_table = pd.read_csv(
        "https://raw.githubusercontent.com/skojaku/core-periphery-detection/master/data/node-table-airport.csv"
    )

    # Edge table
    edge_table = pd.read_csv(
        "https://raw.githubusercontent.com/skojaku/core-periphery-detection/master/data/edge-table-airport.csv"
    )
    # net = nx.adjacency_matrix(nx.from_pandas_edgelist(edge_table))

    net = sparse.csr_matrix(
        (
            edge_table["weight"].values,
            (edge_table["source"].values, edge_table["target"].values),
        ),
        shape=(node_table.shape[0], node_table.shape[0]),
    )

    s = ~pd.isna(node_table["region"])
    node_table = node_table[s]
    labels = node_table["region"].values
    net = net[s, :][:, s]
    return net, labels, node_table


net, labels, node_table = load_airport_net()

# %%
#
# Embedding
#
import torch_geometric
import torch

embedding_models = {}
embedding_model = lambda f: embedding_models.setdefault(f.__name__, f)


def gnn_embedding(model, network, device=None, epochs=50, negative_edge_sampler=None):
    if device is None:
        device = embcom.gnns.get_gpu_id()

    model, emb = embcom.gnns.train(
        model=model,
        feature_vec=None,
        net=network,
        negative_edge_sampler=negative_edge_sampler,
        device=device,
        epochs=epochs,
    )
    return emb


@embedding_model
def dcGIN(network, dim, feature_dim=64, device=None, dim_h=64):
    return gnn_embedding(
        model=torch_geometric.nn.models.GIN(
            in_channels=feature_dim,
            hidden_channels=dim_h,
            num_layers=2,
            out_channels=dim,
        ),
        network=network,
        negative_edge_sampler=embcom.gnns.NegativeEdgeSampler["degreeBiased"],
    )


@embedding_model
def dcPNA(network, dim, feature_dim=64, device=None, dim_h=64):
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
            deg=torch.FloatTensor(np.bincount(np.array(net.sum(axis=0)).reshape(-1))),
        ),
        network=network,
        negative_edge_sampler=embcom.gnns.NegativeEdgeSampler["degreeBiased"],
    )


@embedding_model
def dcEdgeCNN(network, dim, feature_dim=64, device=None, dim_h=64):
    return gnn_embedding(
        model=torch_geometric.nn.models.EdgeCNN(
            in_channels=feature_dim,
            hidden_channels=dim_h,
            num_layers=2,
            out_channels=dim,
        ),
        network=network,
        negative_edge_sampler=embcom.gnns.NegativeEdgeSampler["degreeBiased"],
    )


@embedding_model
def dcGraphUNet(network, dim, feature_dim=64, device=None, dim_h=64):
    return gnn_embedding(
        model=torch_geometric.nn.models.GraphUNet(
            in_channels=feature_dim,
            hidden_channels=dim_h,
            out_channels=dim,
            depth=2,
        ),
        network=network,
        negative_edge_sampler=embcom.gnns.NegativeEdgeSampler["degreeBiased"],
    )


@embedding_model
def GIN(network, dim, feature_dim=64, device=None, dim_h=64):
    return gnn_embedding(
        model=torch_geometric.nn.models.GIN(
            in_channels=feature_dim,
            hidden_channels=dim_h,
            num_layers=2,
            out_channels=dim,
        ),
        network=network,
    )


@embedding_model
def PNA(network, dim, feature_dim=64, device=None, dim_h=64):
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
            deg=torch.FloatTensor(np.bincount(np.array(net.sum(axis=0)).reshape(-1))),
        ),
        network=network,
    )


@embedding_model
def EdgeCNN(network, dim, feature_dim=64, device=None, dim_h=64):
    return gnn_embedding(
        model=torch_geometric.nn.models.EdgeCNN(
            in_channels=feature_dim,
            hidden_channels=dim_h,
            num_layers=2,
            out_channels=dim,
        ),
        network=network,
    )


@embedding_model
def GraphUNet(network, dim, feature_dim=64, device=None, dim_h=64):
    return gnn_embedding(
        model=torch_geometric.nn.models.GraphUNet(
            in_channels=feature_dim,
            hidden_channels=dim_h,
            out_channels=dim,
            depth=2,
        ),
        network=network,
    )


dim = 64
dim_h = 64
feature_dim = 64
for name, emb_model in embedding_models.items():
    print(name)
    emb_model(net, 64)
# gnn = embcom.gnns.GraphSAGE(dim_in=feature_dim, dim_h=dim_h, dim_out=dim)
# gnn = embcom.gnns.train(
#    model=gnn,
#    feature_vec=None,
#    net=net,
#    negative_edge_sampler=embcom.gnns.NegativeEdgeSampler["uniform"],
#    device=device,
#    epochs=1000,
# )
# emb = gnn.generate_embedding(feature_vec=None, net=net, device=device)
# embcom.gnns.
# model.fit(net)
# emb = model.transform(dim=64)

# %%
emb.shape
# %%
# Plot
#
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis(n_components=2)
xy = clf.fit_transform(emb, np.unique(labels, return_inverse=True)[1])


plot_data = pd.DataFrame(
    {
        "x": xy[:, 0],
        "y": xy[:, 1],
        "deg": np.array(net.sum(axis=0)).reshape(-1),
        "label": labels,
    }
)

sns.scatterplot(data=plot_data, x="x", y="y", hue="label", size="deg")


# %%
import torch

r, c, _ = sparse.find(net)
edge_index = torch.LongTensor(np.array([r.astype(int), c.astype(int)]))
edge_index
t = edge_index.clone().reshape(-1)
idx = torch.randperm(t.shape[0])
t = t[idx].view(edge_index.size())
