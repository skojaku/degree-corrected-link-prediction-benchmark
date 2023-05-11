# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-05-10 04:51:58
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-11 05:02:28
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
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader
import GPUtil

#
# Models
#


class GNNBase(torch.nn.Module):
    """A python class for Graph neural networks

    Parameters
    ----------
    dim_in: dimension of in vector
    dim_out: dimension of out vector
    dim_h : dimension of hidden layer

    Example
    ----------
    >> import networkx as nx
    >> G = nx.karate_club_graph()
    >> A = nx.adjacency_matrix(G)
    >> labels = np.unique([d[1]["club"] for d in G.nodes(data=True)], return_inverse=True)[1]
    >> model.fit(A)
    >> gnn_model = GAT(dim_in=4, dim_h=128, dim_out=64)
    >> gnn_model = train(
    >>     model=gnn_model,
    >>     feature_vec=None,
    >>     net=A,
    >>     device="cuda:0",
    >>     epochs=100,
    >> )
    >> emb = gnn_model.generate_embedding(feature_vec=None, net=A, device="cuda:0")
    """

    def __init__(self, dim_in, dim_out):
        """
        Initializes the GNNBase object.

        Parameters
        ----------
        dim_in : int
            The input dimension of the GNN.
        dim_out : int
            The output dimension of the GNN.

        Returns
        -------
        None
        """
        super(GNNBase, self).__init__()
        self.conv1 = None
        self.conv2 = None
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.base_emb = None

    def forward(self, x, positive_edge_index):
        """
        Perform the forward pass through the graph convolutional network.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape (num_nodes, input_dim).
        positive_edge_index : torch.Tensor
            Edge index tensor of shape (2, num_edges). Specifies the indices of the nodes connected by each edge.

        Returns
        -------
        torch.Tensor
            The output node feature matrix of shape (num_nodes, output_dim) after passing through the GCN.
        """
        h = self.conv1(x, positive_edge_index)
        h = h.relu()
        h = self.conv2(h, positive_edge_index)
        return h

    def decode(self, z, pos_edge_index, neg_edge_index):
        """
        Decodes the graph by computing dot products for positive and negative edges.

        Parameters
        ----------
        z (torch.Tensor): Node embeddings of shape [num_nodes, hidden_channels].
        pos_edge_index (torch.Tensor): Tensor of shape [2, num_pos_edges] representing the
            indices of positive edges.
        neg_edge_index (torch.Tensor): Tensor of shape [2, num_neg_edges] representing the
            indices of negative edges.

        Returns
        -------
            logits (torch.Tensor): Tensor of shape [num_pos_edges + num_neg_edges] representing the
                dot product between node embeddings for each edge.
        """
        edge_index = torch.cat(
            [pos_edge_index, neg_edge_index], dim=-1
        )  # concatenate pos and neg edges
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)  # dot product
        return logits

    def generate_embedding(self, net, feature_vec=None, device="cpu"):
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
        edge_index = torch.LongTensor(
            np.array([rows.astype(int), cols.astype(int)])
        ).to(
            device
        )  # Convert the indices to a tensor and move it to the specified device

        if feature_vec is None:
            if self.base_emb is None:
                self.generate_base_embedding(net)
            feature_vec = self.base_emb

        embeddings = self.forward(
            torch.FloatTensor(feature_vec).to(device), edge_index
        )  # Generate embeddings using the model
        return (
            embeddings.detach().cpu().numpy()
        )  # Detach the embeddings from the computation graph and convert it to a numpy array on the CPU

    def generate_base_embedding(self, A):
        """
        Compute the base embedding using the input adjacency matrix.

        Parameters
        ----------
        A (numpy.ndarray): Input adjacency matrix

        Returns
        -------
        numpy.ndarray: Base embedding computed using normalized laplacian matrix
        """
        # Compute the (inverse) normalized laplacian matrix
        deg = np.array(A.sum(axis=1)).reshape(-1)
        Dsqrt = sparse.diags(1 / np.maximum(np.sqrt(deg), 1e-12), format="csr")
        L = Dsqrt @ A @ Dsqrt
        L.setdiag(1)

        s, u = sparse.linalg.eigs(L, k=self.dim_in, which="LR")
        s, u = np.real(s), np.real(u)
        order = np.argsort(-s)[1:]
        s, u = s[order], u[:, order]
        Dsqrt = sparse.diags(1 / np.maximum(np.sqrt(deg), 1e-12), format="csr")
        base_emb = Dsqrt @ u @ sparse.diags(np.sqrt(np.abs(s)))
        mean_norm = np.mean(np.linalg.norm(base_emb, axis=0))
        _deg = deg / np.sum(deg)
        _deg = mean_norm * _deg / np.linalg.norm(_deg)
        base_emb = np.hstack([base_emb, _deg.reshape((-1, 1))])
        self.base_emb = base_emb
        return base_emb


class GCN:
    """
    A Graph Convolutional Network (GCN) implemented using PyTorch.

    Parameters
    ----------
    dim_in : int
        The number of input features.
    dim_h : int
        The number of hidden units.
    dim_out : int
        The number of output features.

    Attributes
    ----------
    conv1 : GCNConv
        The first graph convolutional layer.
    conv2 : GCNConv
        The second graph convolutional layer.
    """

    def __init__(self, dim_in, dim_h, dim_out):
        """
        Initializes a new instance of the GCN class.

        Parameters
        ----------
        dim_in : int
            The number of input features.
        dim_h : int
            The number of hidden units.
        dim_out : int
            The number of output features.
        """
        super(GCN, self).__init__(dim_in, dim_out)
        self.conv1 = GCNConv(dim_in, dim_h)
        self.conv2 = GCNConv(dim_h, dim_out)


class GraphSAGE(GNNBase):
    """
    Implementation of the GraphSAGE model for graph neural networks.

    Parameters:
    -----------
    dim_in : int
        Number of input features.
    dim_h : int
        Number of hidden features.
    dim_out : int
        Number of output features.

    Attributes:
    -----------
    conv1 : SAGEConv
        First GraphSAGE convolutional layer.
    conv2 : SAGEConv
        Second GraphSAGE convolutional layer.

    Returns:
    --------
    None
    """

    def __init__(self, dim_in, dim_h, dim_out):
        super(GraphSAGE, self).__init__(dim_in, dim_out)
        self.conv1 = SAGEConv(dim_in, dim_h, project=True, aggr="max")
        self.conv2 = SAGEConv(dim_h, dim_out)


class GAT(GNNBase):
    """
    Implementation of the GAT model for graph neural networks.

    Parameters:
    -----------
    dim_in : int
        Number of input features.
    dim_h : int
        Number of hidden features.
    dim_out : int
        Number of output features.

    Attributes:
    -----------
    conv1 : GATConv
        First GAT convolutional layer.
    conv2 : GATConv
        Second GAT convolutional layer.

    Returns:
    --------
    None
    """

    def __init__(self, dim_in, dim_h, dim_out):
        super(GAT, self).__init__(dim_in, dim_out)
        self.conv1 = GATConv(dim_in, dim_h, dropout=0.2)
        self.conv2 = GATConv(dim_h, dim_out, dropout=0.2)


#
# negative edge sampling
#
def degreeBiasedNegativeEdgeSampling(edge_index, num_nodes, num_neg_samples):
    t = edge_index.clone().reshape(-1)
    idx = torch.randperm(t.shape[0])
    t = t[idx].view(edge_index.size())
    return t


NegativeEdgeSampler = {
    "degreeBiased": degreeBiasedNegativeEdgeSampling,
    "uniform": negative_sampling,
}


#
# Utilities
#
def train(
    model: torch.nn.Module,
    feature_vec: np.ndarray,
    net: sparse.spmatrix,
    device: str,
    epochs: int,
    negative_edge_sampler=None,
    batch_size: int = 2500,
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
        feature_vec = model.generate_base_embedding(net)

    data = Data(x=torch.FloatTensor(feature_vec), edge_index=edge_index)

    # Move the model to the specified device
    model.to(device)

    # Set up minibatching for the data using a clustering algorithm
    n_nodes = net.shape[0]
    num_sub_batches = 5
    batch_size = np.minimum(n_nodes, batch_size)
    cluster_data = ClusterData(
        data, num_parts=int(np.floor(n_nodes / batch_size) * num_sub_batches)
    )  # 1. Create subgraphs.
    train_loader = ClusterLoader(
        cluster_data, batch_size=num_sub_batches, shuffle=True
    )  # 2. Stochastic partioning scheme.

    # Use default negative sampling function if none is specified
    if negative_edge_sampler is None:
        negative_edge_sampler = negative_sampling

    # Set the model in training mode and initialize optimizer
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train the model for the specified number of epochs
    for epoch in tqdm(range(epochs)):
        # Iterate over minibatches of the data
        for sub_data in train_loader:
            # Sample negative edges using specified or default sampler
            pos_edge_index = sub_data.edge_index  # positive edges
            node_feature_vec = sub_data.x
            neg_edge_index = negative_edge_sampler(
                edge_index=pos_edge_index,
                num_nodes=node_feature_vec.shape[0],
                num_neg_samples=pos_edge_index.size(1),
            )
            neg_edge_index = neg_edge_index.to(device)
            pos_edge_index = pos_edge_index.to(device)
            node_feature_vec = node_feature_vec.to(device)

            # Zero-out gradient, compute embeddings and logits, and calculate loss
            optimizer.zero_grad()
            z = model(node_feature_vec, pos_edge_index)
            link_logits = model.decode(z, pos_edge_index, neg_edge_index)
            link_labels = torch.zeros(
                pos_edge_index.size(1) + neg_edge_index.size(1),
                dtype=torch.float,
                device=device,
            )
            link_labels[: pos_edge_index.size(1)] = 1.0
            loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)

            # Compute gradients and update parameters of the model
            loss.backward()
            optimizer.step()

    # Set the model in evaluation mode and return
    model.eval()
    return model


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
