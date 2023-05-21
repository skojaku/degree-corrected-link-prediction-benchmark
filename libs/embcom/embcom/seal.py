# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-05-20 05:50:31
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-20 21:01:26
from tqdm.auto import tqdm
import scipy
from numba import njit
import os.path as osp
import sys
import numpy as np
from scipy import sparse
import torch
import torch.utils.data
from scipy.sparse.csgraph import shortest_path
import utils
from torch_geometric.loader import DataLoader


class SEALUtils:
    def generate_input_data(src, trg, x, net, hops, max_nodes_per_hop):
        nodes = SEALUtils.get_enclosing_subgraph(
            src,
            trg,
            hops=hops,
            net=net,
            max_nodes_per_hop=max_nodes_per_hop,
        )

        subnet = net[nodes, :][:, nodes]
        sub_x = x[nodes, :]

        # code from here. labelling
        node_labels = SEALUtils.dual_radius_node_labelling(subnet)
        sub_x = torch.cat(
            [sub_x, torch.FloatTensor(node_labels).reshape((-1, 1))], dim=-1
        )
        return subnet, sub_x

    def dual_radius_node_labelling(subnet):
        # Function to label nodes in a graph based on their distance from two selected nodes.
        node_labels = np.zeros(subnet.shape[0])

        # Compute shortest path between the first two nodes.
        dist = shortest_path(csgraph=subnet, directed=False, indices=[0, 1]).astype(int)

        # Compute the sum of distances from all nodes to these two nodes.
        d = np.array(np.sum(dist, axis=0)).reshape(-1)

        # Compute minimum distance of each node from the two nodes.
        dmin = np.minimum(dist[0], dist[1])

        # Assign labels to nodes according to their distance from the two chosen nodes.
        node_labels = 1 + dmin + (d // 2) * (d // 2 + d % 2 - 1)
        node_labels[0] = 1
        node_labels[1] = 1

        # Handle cases where there is no path between a node and one of the two selected nodes.
        node_labels[np.isinf(node_labels)] = 0
        node_labels[np.isnan(node_labels)] = 0
        return node_labels

    def get_enclosing_subgraph(src, trg, hops, net, max_nodes_per_hop=None):
        # Function to return a subgraph containing source and target nodes and their neighbors within a certain number of hops.
        if max_nodes_per_hop is None:
            max_nodes_per_hop = net.shape[0]

        nodes = set([src, trg])
        visited = nodes.copy()

        for _ in range(hops):
            _, neighbors, v = sparse.find(net[np.array(list(nodes)), :].sum(axis=0))

            # Limit the number of neighbors to be considered.
            if len(neighbors) > max_nodes_per_hop:
                neighbors = np.random.choice(
                    neighbors, size=max_nodes_per_hop, replace=False
                )
            visited.update(neighbors)
            nodes = neighbors

        visited = np.sort(np.array(list(visited)))

        # Ensure that the source and target nodes are included in the subgraph.
        i = np.searchsorted(visited, src)
        j = np.searchsorted(visited, trg)
        if i > 1:
            visited[np.array([i, 0])] = visited[np.array([0, i])]
        if j > 1:
            visited[np.array([j, 1])] = visited[np.array([1, j])]
        return visited.astype(int)


class SEALDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        edge_index,
        x,
        n_nodes,
        negative_edge_sampler,
        hops=2,
        max_nodes_per_hop=100,
    ):
        # Constructor for SEALClusterData class. Initializes instance variables.

        # Convert edge_index to a LongTensor and remove duplicate edges
        undirected_edge_index = torch.LongTensor(
            np.vstack(
                [
                    *utils.depairing(
                        np.unique(utils.pairing(edge_index[0], edge_index[1]))
                    )
                ]
            )
        )

        # Set instance variables
        self.n_nodes = n_nodes
        self.pos_edge_index = undirected_edge_index
        self.x = x

        # Sample negative edges using the given sampler
        self.neg_edge_index = utils.sample_unconnected_node_pairs(
            self.pos_edge_index,
            self.n_nodes,
            num_samples=self.pos_edge_index.size()[1],
            sampler=negative_edge_sampler,
        )
        self.negative_edge_sampler = negative_edge_sampler
        self.neg_edge_index = torch.LongTensor(self.neg_edge_index)

        # Concatenate positive and negative edges and set related instance variables
        self.edge_index = torch.cat([self.pos_edge_index, self.neg_edge_index], dim=-1)
        self.n_pos_edges = self.pos_edge_index.shape[1]
        self.n_neg_edges = self.neg_edge_index.shape[1]
        self.n_edges = self.n_pos_edges + self.n_neg_edges

        # Create target labels (1 for positive edges, 0 for negative edges)
        self.y = torch.cat(
            [
                torch.ones(self.n_pos_edges, dtype=torch.long),
                torch.zeros(self.n_neg_edges, dtype=torch.long),
            ]
        )

        # Create a sparse adjacency matrix for the positive edges
        rows, cols = self.pos_edge_index[0], self.pos_edge_index[1]
        rows, cols = np.concatenate([rows, cols]), np.concatenate([cols, rows])
        self.net = sparse.csr_matrix(
            (np.ones_like(rows), (rows, cols)), shape=(self.n_nodes, self.n_nodes)
        )

        # Initialize variables with given inputs
        self.hops = hops
        self.max_nodes_per_hop = max_nodes_per_hop

    def __len__(self):
        # Returns the number of edges (positive + negative) in the dataset
        return self.n_edges

    def __getitem__(self, idx):
        # Function to get a subgraph with nodes surrounding a randomly selected edge

        # Get target label and edge index at given index
        y = self.y[idx]
        edge = self.edge_index[:, idx]

        # Generate input data for subgraph surrounding the selected edge
        subnet, sub_x = SEALUtils.generate_input_data(
            src=edge[0],
            trg=edge[1],
            x=self.x,
            net=self.net,
            hops=self.hops,
            max_nodes_per_hop=self.max_nodes_per_hop,
        )

        # Return dictionary containing the target label, input features, adjacency matrix, and edge index
        return {
            "y": y,
            "x": sub_x,
            "net": subnet,
            "edge": edge,
        }

    def regenerate(self):
        # Sample negative edges using the given sampler
        self.neg_edge_index = utils.sample_unconnected_node_pairs(
            self.pos_edge_index,
            self.n_nodes,
            num_samples=self.pos_edge_index.size()[1],
            sampler=self.negative_edge_sampler,
        )
        self.neg_edge_index = torch.LongTensor(self.neg_edge_index)

        # Concatenate positive and negative edges and set related instance variables
        # self.edge_index[:, self.n_pos_edges :] = self.neg_edge_index

    def __repr__(self):
        # Returns a string representation of the SEALDataset object
        return (
            f"{self.__class__.__name__}(\n"
            f"  hops={self.hops},\n"
            f"  n_edges={self.n_edges}\n"
            f")"
        )


class SEALDataloader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, **params):
        # Define collate function to convert list of samples to batch tensor dictionary
        collate_fn = lambda data: {
            "y": torch.FloatTensor([d["y"] for d in data]),
            "x": torch.vstack([d["x"] for d in data]),  # stack x features vertically
            "edge_index": torch.LongTensor(
                utils.adj2edgeindex(
                    scipy.sparse.block_diag([d["net"] for d in data])
                )  # convert adjacency matrices to edge indices
            ),
            "edge": torch.vstack([d["edge"] for d in data]).T,
            "block_ptr": torch.LongTensor(
                np.insert(np.cumsum([d["net"].shape[0] for d in data]), 0, 0)[
                    :-1
                ]  # pointer to indicate the start and end of each block in the edge_index tensor
            ),
        }
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            **params,  # pass any additional DataLoader parameters to the parent class constructor
        )


def SEALTrain(
    model: torch.nn.Module,
    feature_vec: np.ndarray,
    net: sparse.spmatrix,
    device: str,
    epochs: int,
    hops: int = 2,
    feature_vec_dim: int = 64,
    negative_edge_sampler=None,
    batch_size: int = 50,
    lr=0.01,
) -> torch.nn.Module:
    n_nodes = net.shape[0]

    # Convert sparse adjacency matrix to edge list format
    r, c, _ = sparse.find(net)
    edge_index = torch.LongTensor(np.array([r.astype(int), c.astype(int)]))

    # If feature vectors are not provided, generate them using the default method
    if feature_vec is None:
        feature_vec = gnns.generate_base_embedding(net, feature_vec_dim)
        feature_vec = torch.FloatTensor(feature_vec)

    # Use default negative sampling function if none is specified
    if negative_edge_sampler is None:
        negative_edge_sampler = negative_uniform

    # Create PyTorch data object with features and edge list
    data = SEALDataset(
        edge_index=edge_index,
        x=feature_vec,
        n_nodes=net.shape[0],
        hops=hops,
        negative_edge_sampler=negative_edge_sampler,
    )
    train_loader = SEALDataloader(data, batch_size=batch_size, shuffle=True)

    # Set the model in training mode and initialize optimizer
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model for the specified number of epochs
    pbar = tqdm(total=epochs)  # create a progress bar to display during training
    loss_func = torch.nn.BCEWithLogitsLoss()  # use binary cross-entropy loss
    for epoch in range(epochs):
        if epochs != 0:
            data.regenerate()
            train_loader = SEALDataloader(data, batch_size=batch_size, shuffle=True)

        ave_loss = 0
        n_iter = 0
        # Iterate over minibatches of the data
        for batch in train_loader:
            # Zero-out gradient, compute embeddings and logits, and calculate loss
            optimizer.zero_grad()

            # Move batch tensors to the device (CPU/GPU) being used for computation
            for k, v in batch.items():
                batch[k] = v.to(device)

            # Generate node embeddings for each block in the batch and calculate logits
            emb = model(batch["x"], batch["edge_index"])

            score = (emb[batch["block_ptr"], :] * emb[batch["block_ptr"] + 1, :]).sum(
                dim=1
            )
            # Compute binary cross-entropy loss and backpropagate gradients
            loss = loss_func(score, batch["y"])
            loss.backward()
            optimizer.step()

            # Compute average loss over the entire dataset
            with torch.no_grad():
                ave_loss += loss.item()
                n_iter += 1
        # Update progress bar and display current loss and number of iterations per epoch

        pbar.update(1)
        ave_loss /= n_iter
        pbar.set_description(f"loss={ave_loss:.3f} iter/epoch={n_iter}")

    # Set the model in evaluation mode and return
    model.eval()
    return model


#
#import networkx as nx
#from scipy.sparse.csgraph import shortest_path
#from scipy import sparse
#import numpy as np
#from torch_geometric.data import Data
#import gnns
#from torch.utils.data import DataLoader
#
#G = nx.karate_club_graph()
#net = nx.adjacency_matrix(G)
## labels = np.unique([d[1]['club'] for d in G.nodes(data=True)], return_inverse=True)[1]
#net[22, 33] = 0
#net[33, 22] = 0
#net = sparse.csr_matrix(net)
#net.eliminate_zeros()
#net.data = net.data * 0 + 1
#
#
#def negative_uniform(edge_index, num_nodes, num_neg_samples):
#    t = np.random.randint(
#        0, num_nodes, size=num_neg_samples * edge_index.size()[1]
#    ).reshape((num_neg_samples, edge_index.size()[1]))
#    return torch.LongTensor(t)
#
#
#def degreeBiasedNegativeEdgeSampling(edge_index, num_nodes, num_neg_samples):
#    deg = np.bincount(edge_index.reshape(-1).cpu(), minlength=num_nodes).astype(float)
#    deg /= np.sum(deg)
#    t = np.random.choice(
#        num_nodes, p=deg, size=num_neg_samples * edge_index.size()[1]
#    ).reshape((num_neg_samples, edge_index.size()[1]))
#    return torch.LongTensor(t)
#
#
#n_nodes = net.shape[0]
#
## Convert sparse adjacency matrix to edge list format
#r, c, _ = sparse.find(net)
#edge_index = torch.LongTensor(np.array([r.astype(int), c.astype(int)]))
#
#feature_vec = gnns.generate_base_embedding(net, 16)
#feature_vec = torch.FloatTensor(feature_vec)
#
#import torch_geometric
#
#feature_dim = feature_vec.shape[1] + 1
#dim_h = 64
#dim = 64
#gnn_model = torch_geometric.nn.models.GCN(
#    in_channels=feature_dim,
#    hidden_channels=dim_h,
#    num_layers=2,
#    out_channels=dim,
#)
#
#gnn_model = SEALTrain(
#    model=gnn_model,
#    feature_vec=feature_vec,
#    net=net,
#    device="cuda:1",
#    epochs=100,
#    hops=2,
#    feature_vec_dim=64,
#    #negative_edge_sampler=degreeBiasedNegativeEdgeSampling,
#    negative_edge_sampler=negative_uniform,
#    batch_size=50,
#    lr=0.01,
#)
#
#
#gnn_model.to("cpu")
#P = np.zeros(net.shape)
#for src in range(net.shape[0]):
#    for trg in range(src, net.shape[0]):
#        if net[src, trg] != 0:
#            continue
#        subnet, sub_x = SEALUtils.generate_input_data(
#            src, trg, feature_vec, net, hops=2, max_nodes_per_hop=None
#        )
#        sub_edge_index = torch.LongTensor(utils.adj2edgeindex(subnet))
#        emb = gnn_model(sub_x, sub_edge_index)
#        emb = emb.detach().cpu().numpy()
#        P[src, trg] = np.sum(emb[0, :] * emb[1, :])
#        P[trg, src] = P[src, trg]
#import seaborn as sns
#import matplotlib.pyplot as plt
#
#sns.heatmap(P, cmap="coolwarm", center=0)
## %%
## np.where(np.max(P) == P)
#P[22, :]
#