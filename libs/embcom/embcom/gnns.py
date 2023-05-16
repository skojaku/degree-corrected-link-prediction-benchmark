# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-05-10 04:51:58
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-15 16:47:34
# %%
import numpy as np
from scipy import sparse
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
import torch
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric import nn

from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader
import GPUtil


#
# Models
#
def generate_embedding(model, net, feature_vec=None, device="cpu"):
    """Generate embeddings using a specified network model.

    Parameters
    ----------
        net (sparse matrix): The input sparse matrix.
        feature_vec (ndarray, optional): The initial feature vector. If not provided,
            the base embedding will be used instead. Defaults to None.
        device (str, optional): The device to use for computations. Defaults to "cpu".

    Returns
    -------
        ndarray: The resulting embeddings as a numpy array on the CPU.
    """
    rows, cols, _ = sparse.find(
        net
    )  # Find the row and column indices of non-zero elements in the sparse matrix
    edge_index = torch.LongTensor(np.array([rows.astype(int), cols.astype(int)])).to(
        device
    )  # Convert the indices to a tensor and move it to the specified device

    if feature_vec is None:
        feature_vec = generate_base_embedding(net, dim=model.in_channels)

    embeddings = model.forward(
        torch.FloatTensor(feature_vec).to(device), edge_index
    )  # Generate embeddings using the model
    return (
        embeddings.detach().cpu().numpy()
    )  # Detach the embeddings from the computation graph and convert it to a numpy array on the CPU


from sklearn.decomposition import TruncatedSVD


def generate_base_embedding(A, dim):
    """
    Compute the base embedding using the input adjacency matrix.

    Parameters
    ----------
    A (numpy.ndarray): Input adjacency matrix

    Returns
    -------
    numpy.ndarray: Base embedding computed using normalized laplacian matrix
    """
    #    svd = TruncatedSVD(n_components=dim, n_iter=7, random_state=42)
    #    base_emb = svd.fit_transform(A)
    #    base_emb = np.einsum("ij,j->ij", base_emb, 1 / np.linalg.norm(base_emb, axis=0))
    #    return base_emb

    # Compute the (inverse) normalized laplacian matrix

    deg = np.array(A.sum(axis=1)).reshape(-1)
    Dsqrt = sparse.diags(1 / np.maximum(np.sqrt(deg), 1e-12), format="csr")
    L = Dsqrt @ A @ Dsqrt
    L.setdiag(1)

    s, u = sparse.linalg.eigs(L, k=dim, which="LR")
    s, u = np.real(s), np.real(u)
    order = np.argsort(-s)[1:]
    s, u = s[order], u[:, order]
    Dsqrt = sparse.diags(1 / np.maximum(np.sqrt(deg), 1e-12), format="csr")
    base_base_emb = Dsqrt @ u @ sparse.diags(np.sqrt(np.abs(s)))
    mean_norm = np.mean(np.linalg.norm(base_base_emb, axis=0))
    _deg = np.log(deg)
    # / np.sum(deg)
    # _deg = mean_norm * _deg / np.linalg.norm(_deg)
    base_base_emb = np.hstack([base_base_emb, _deg.reshape((-1, 1))])
    # base_base_emb = np.einsum(
    #    "ij,j->ij", base_base_emb, 1 / np.linalg.norm(base_base_emb, axis=0)
    # )
    return base_base_emb


#
# negative edge sampling
#
def degreeBiasedNegativeEdgeSampling(edge_index, num_nodes, num_neg_samples):
    deg = np.bincount(edge_index.reshape(-1).cpu(), minlength=num_nodes).astype(float)
    deg /= np.sum(deg)
    t = np.random.choice(
        num_nodes, p=deg, size=num_neg_samples * edge_index.size()[1]
    ).reshape((num_neg_samples, edge_index.size()[1]))
    return torch.LongTensor(t)


def negative_uniform(edge_index, num_nodes, num_neg_samples):
    t = np.random.randint(
        0, num_nodes, size=num_neg_samples * edge_index.size()[1]
    ).reshape((num_neg_samples, edge_index.size()[1]))
    return torch.LongTensor(t)


NegativeEdgeSampler = {
    "degreeBiased": degreeBiasedNegativeEdgeSampling,
    "uniform": negative_uniform,
}


class GNNwithEmbLayer(torch.nn.Module):
    def __init__(self, gnn, n_nodes, dim_emb, feature_vec=None):
        super(GNNwithEmbLayer, self).__init__()
        self.gnn = gnn
        self.emb_layer = torch.nn.Embedding(n_nodes, dim_emb, padding_idx=-1)
        if feature_vec is None:
            self.emb_layer.weight = torch.nn.Parameter(
                torch.FloatTensor(n_nodes, dim_emb).uniform_(
                    -0.5 / dim_emb, 0.5 / dim_emb
                )
            )
            self.emb_layer.weight.requires_grad = True
        else:
            self.emb_layer.weight = torch.nn.Parameter(torch.FloatTensor(feature_vec))
            self.emb_layer.weight.requires_grad = False
        self.n_nodes = n_nodes

    def forward(self, edge_index, node_ids=None):
        if node_ids is None:
            node_ids = torch.arange(self.n_nodes, device=self.emb_layer.weight.device)
        base_emb = self.emb_layer(node_ids)
        return self.gnn(base_emb, edge_index)


#
# Utilities
#
def train(
    model: torch.nn.Module,
    feature_vec: np.ndarray,
    net: sparse.spmatrix,
    device: str,
    epochs: int,
    feature_vec_dim: int = 64,
    negative_edge_sampler=None,
    batch_size: int = 2500,
    lr=0.01,
) -> torch.nn.Module:
    """
    Train a PyTorch model on a given graph dataset using minibatch stochastic gradient descent with negative sampling.

    Parameters
    ----------
    model : nn.Module
        A PyTorch module representing the model to be trained.
    feature_vec : np.ndarray
        A numpy array of shape (num_nodes, num_features) containing the node feature vectors for the graph.
    net : sp.spmatrix
        A scipy sparse matrix representing the adjacency matrix of the graph.
    device : str
        The device to use for training the model.
    epochs : int
        The number of epochs to train the model for.
    negative_edge_sampler : Callable, optional
        A function that samples negative edges given positive edges and the number of nodes in the graph.
        If unspecified, a default negative sampling function is used.
    batch_size : int, optional
        The number of nodes in each minibatch.

    Returns
    -------
    nn.Module
        The trained model.
    """
    n_nodes = net.shape[0]

    # Convert sparse adjacency matrix to edge list format
    r, c, _ = sparse.find(net)
    edge_index = torch.LongTensor(np.array([r.astype(int), c.astype(int)]))

    # Create PyTorch data object with features and edge list
    data = Data(edge_index=edge_index, node_ids=torch.arange(n_nodes))

    # Move the model to the specified device
    model.to(device)

    # Set up minibatching for the data using a clustering algorithm
    num_sub_batches = 5
    batch_size = np.minimum(n_nodes, batch_size)
    num_parts = int(np.floor(n_nodes / batch_size))
    cluster_data = ClusterData(data, num_parts=num_parts)  # 1. Create subgraphs.
    train_loader = ClusterLoader(
        cluster_data, batch_size=num_sub_batches, shuffle=False
    )  # 2. Stochastic partioning scheme.

    # Use default negative sampling function if none is specified
    if negative_edge_sampler is None:
        negative_edge_sampler = negative_uniform
        # snegative_edge_sampler = negative_sampling

    # Set the model in training mode and initialize optimizer
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Train the model for the specified number of epochs
    pbar = tqdm(total=epochs)
    logsigmoid = torch.nn.LogSigmoid()

    for epoch in range(epochs):
        # Iterate over minibatches of the data
        ave_loss = 0
        n_iter = 0
        for sub_data in train_loader:
            # Sample negative edges using specified or default sampler
            pos_edge_index = sub_data.edge_index  # positive edges
            node_ids = sub_data.node_ids
            neg_edge_index = negative_edge_sampler(
                edge_index=pos_edge_index,
                num_nodes=len(node_ids),
                num_neg_samples=2,
            )

            node_ids = node_ids.to(device)
            neg_edge_index = neg_edge_index.to(device)
            pos_edge_index = pos_edge_index.to(device)

            # Zero-out gradient, compute embeddings and logits, and calculate loss
            optimizer.zero_grad()
            z = model(
                torch.cat([pos_edge_index, neg_edge_index], dim=-1), node_ids=node_ids
            )
            # z = model(pos_edge_index, node_ids=node_ids)
            # z = model(pos_edge_index, node_ids=node_ids)

            pos = (z[pos_edge_index[0], :] * z[pos_edge_index[1], :]).sum(dim=1)
            ploss = -pos.mean()
            ploss = -logsigmoid(pos).mean()
            nloss = 0
            #            for i in range(neg_edge_index.size()[0]):
            #                neg = (z[pos_edge_index[0], :] * z[neg_edge_index[i], :]).sum(dim=1)
            #                nloss -= logsigmoid(neg.neg()).mean()
            # for i in range(neg_edge_index.size()[0]):
            neg = (z[neg_edge_index[0], :] * z[neg_edge_index[1], :]).sum(dim=1)
            nloss -= logsigmoid(neg.neg()).mean()
            loss = ploss + nloss / neg_edge_index.size()[0]

            # Compute gradients and update parameters of the model
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                ave_loss += loss.item()
                n_iter += 1
        pbar.update(1)
        ave_loss /= n_iter
        pbar.set_description(f"loss={ave_loss:.3f} iter/epoch={n_iter}")

    # Set the model in evaluation mode and return
    model.eval()
    emb = model(edge_index.to(device))
    emb = emb.detach().cpu().numpy()
    return model, emb


def train_all(
    model: torch.nn.Module,
    feature_vec: np.ndarray,
    net: sparse.spmatrix,
    device: str,
    epochs: int,
    feature_vec_dim: int = 64,
    negative_edge_sampler=None,
    lr=0.01,
) -> torch.nn.Module:
    """
    Train a PyTorch model on a given graph dataset using minibatch stochastic gradient descent with negative sampling.

    Parameters
    ----------
    model : nn.Module
        A PyTorch module representing the model to be trained.
    feature_vec : np.ndarray
        A numpy array of shape (num_nodes, num_features) containing the node feature vectors for the graph.
    net : sp.spmatrix
        A scipy sparse matrix representing the adjacency matrix of the graph.
    device : str
        The device to use for training the model.
    epochs : int
        The number of epochs to train the model for.
    negative_edge_sampler : Callable, optional
        A function that samples negative edges given positive edges and the number of nodes in the graph.
        If unspecified, a default negative sampling function is used.
    batch_size : int, optional
        The number of nodes in each minibatch.

    Returns
    -------
    nn.Module
        The trained model.
    """

    # Convert sparse adjacency matrix to edge list format
    r, c, _ = sparse.find(net)
    edge_index = torch.LongTensor(np.array([r.astype(int), c.astype(int)]))

    # Create PyTorch data object with features and edge list
    if feature_vec is None:
        feature_vec = generate_base_embedding(net, dim=feature_vec_dim)
    feature_vec = torch.FloatTensor(feature_vec)

    # Set up minibatching for the data using a clustering algorithm
    if negative_edge_sampler is None:
        # negative_edge_sampler = negative_sampling
        negative_edge_sampler = negative_uniform

    # Set the model in training mode and initialize optimizer
    # Move the model to the specified device
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Train the model for the specified number of epochs
    pbar = tqdm(total=epochs)
    logsigmoid = torch.nn.LogSigmoid()
    edge_index = edge_index.to(device)
    feature_vec = feature_vec.to(device)
    for epoch in range(epochs):
        # Iterate over minibatches of the data
        ave_loss = 0
        n_iter = 0
        neg_edge_index = negative_edge_sampler(
            edge_index=edge_index,
            num_nodes=feature_vec.shape[0],
            num_neg_samples=2,
        )
        neg_edge_index = neg_edge_index.to(device)

        # Zero-out gradient, compute embeddings and logits, and calculate loss
        optimizer.zero_grad()
        zpos = model(np.cat([edge_index, neg_edge_index]))
        # zpos = model(edge_index)

        pos = (zpos[edge_index[0], :] * zpos[edge_index[1], :]).sum(dim=1)
        ploss = -pos.mean()
        ploss = -logsigmoid(pos).mean()
        nloss = 0
        for i in range(neg_edge_index.size()[0]):
            neg = (zpos[edge_index[0], :] * zpos[neg_edge_index[i], :]).sum(dim=1)
            nloss -= logsigmoid(neg.neg()).mean()
        loss = ploss + nloss / neg_edge_index.size()[0]

        # Compute gradients and update parameters of the model
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            ave_loss += loss.item()
            n_iter += 1
        pbar.update(1)
        ave_loss /= n_iter
        pbar.set_description(f"loss={ave_loss:.3f} iter/epoch={n_iter}")

    # Set the model in evaluation mode and return
    model.eval()
    emb = model(edge_index.to(device))
    emb = emb.detach().cpu().numpy()
    return model, emb


def get_gpu_id(excludeID=[]):
    device = GPUtil.getFirstAvailable(
        order="random",
        maxLoad=1,
        maxMemory=0.3,
        attempts=99999,
        interval=60 * 1,
        verbose=False,
        excludeID=excludeID,
    )[0]
    device = f"cuda:{device}"
    return device
