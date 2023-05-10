# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-05-10 04:51:58
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-10 04:59:45
import numpy as np
from scipy import sparse
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
import torch
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class GCN(torch.nn.Module):
    """A python class for the GCN.

    Parameters
    ----------
    dim_in: dimension of in vector
    dim_out: dimension of out vector
    dim_h : dimension of hidden layer

    """

    def __init__(self, dim_in, dim_h, dim_out):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dim_in, dim_h)
        self.conv2 = GCNConv(dim_h, dim_out)

    def forward(self, x, positive_edge_index):
        h = self.conv1(x, positive_edge_index)
        h = h.relu()
        h = self.conv2(h, positive_edge_index)
        return h

    def decode(self, z, pos_edge_index, neg_edge_index):  # only pos and neg edges
        edge_index = torch.cat(
            [pos_edge_index, neg_edge_index], dim=-1
        )  # concatenate pos and neg edges
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)  # dot product
        return logits


def get_link_labels(pos_edge_index, neg_edge_index, device):
    # returns a tensor:
    # [1,1,1,1,...,0,0,0,0,0,..] with the number of ones is equel to the lenght of pos_edge_index
    # and the number of zeros is equal to the length of neg_edge_index
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[: pos_edge_index.size(1)] = 1.0
    return link_labels


def train(model, data, adjacency, device, epochs=1000, negative_edge_sampler=None):
    #
    # Create the dataset
    #

    # To edge list
    adj_c = adjacency.tocoo()
    edge_list_torch = torch.from_numpy(np.array([adj_c.row, adj_c.col])).to(device)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in tqdm(range(epochs + 1)):
        if negative_edge_sampler is None:
            neg_edge_index = negative_sampling(
                edge_index=edge_list_torch,  # positive edges
                num_nodes=data.shape[0],  # number of nodes
                num_neg_samples=edge_list_torch.size(1),
            )
        else:
            neg_edge_index = negative_edge_sampler(
                edge_index=edge_list_torch,  # positive edges
                num_nodes=data.shape[0],  # number of nodes
                num_neg_samples=edge_list_torch.size(1),
            )

        optimizer.zero_grad()

        z = model(data, edge_list_torch)

        link_logits = model.decode(z, edge_list_torch, neg_edge_index)  # decode
        link_labels = get_link_labels(edge_list_torch, neg_edge_index, device)
        loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
        loss.backward()
        optimizer.step()

    return model


# import stellargraph
# from tensorflow import keras
# from stellargraph.data import UnsupervisedSampler
# from stellargraph.mapper import GraphSAGELinkGenerator, GraphSAGENodeGenerator
# class graphSAGE:
#    """A python class for the GraphSAGE.
#    Parameters
#    ----------
#    num_walks : int (optional, default 1)
#        Number of walks per node
#    walk_length : int (optional, default 5)
#        Length of walks
#    """
#
#    def __init__(
#        self,
#        num_walks=1,
#        walk_length=5,
#        emb_dim=50,
#        num_samples=[10, 5],
#        batch_size=50,
#        epochs=4,
#        verbose=False,
#        feature_dim=50,
#    ):
#        self.in_vec = None  # In-vector
#        self.out_vec = None  # Out-vector
#        self.feature_vector = None
#        self.model = None
#        self.generator = None
#        self.train_gen = None
#        self.G = None
#
#        self.num_walks = num_walks
#        self.walk_length = walk_length
#        self.layer_sizes = [50, emb_dim]
#        self.num_samples = num_samples
#        self.batch_size = batch_size
#        self.epochs = epochs
#
#        self.verbose = verbose
#        self.feature_dim = feature_dim
#
#    def fit(self, net):
#        """Takes a network as an input, transforms it into an adjacency matrix, and generates
#        feature vectors for nodes and creates a StellarGraph object"""
#
#        # transform into an adjacency matrix
#        A = utils.to_adjacency_matrix(net)
#
#        # create node features
#        self.feature_vector = self.create_feature_vector(
#            A, feature_dim=self.feature_dim
#        )
#
#        # transform the adjacency matrix into a networkx Graph object
#        self.G = nx.Graph(A)
#        nodes = [*self.G.nodes()]
#
#        # transform it into a StellarGraph
#        self.G = stellargraph.StellarGraph.from_networkx(
#            graph=self.G, node_features=zip(nodes, self.feature_vector)
#        )
#
#        # Create the UnsupervisedSampler instance with the relevant parameters passed to it
#        unsupervised_samples = UnsupervisedSampler(
#            self.G, nodes=nodes, length=self.walk_length, number_of_walks=self.num_walks
#        )
#
#        self.generator = GraphSAGELinkGenerator(
#            self.G, self.batch_size, self.num_samples
#        )
#        self.train_gen = self.generator.flow(unsupervised_samples)
#
#        return self
#
#    def create_feature_vector(self, A, feature_dim):
#        """Takes an adjacency matrix and generates feature vectors using
#        Laplacian Eigen Map"""
#        lapeigen = LaplacianEigenMap(p=100, q=40)
#        lapeigen.fit(A)
#        return lapeigen.transform(self.feature_dim, return_out_vector=False)
#
#    def train_GraphSAGE(self):
#        graphsage = GraphSAGE(
#            layer_sizes=self.layer_sizes,
#            generator=self.generator,
#            bias=True,
#            dropout=0.0,
#            normalize="l2",
#        )
#
#        self.in_vec, self.out_vec = graphsage.in_out_tensors()
#
#        prediction = link_classification(
#            output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
#        )(self.out_vec)
#
#        self.model = keras.Model(inputs=self.in_vec, outputs=prediction)
#
#        self.model.compile(
#            optimizer=keras.optimizers.Adam(lr=1e-3),
#            loss=keras.losses.binary_crossentropy,
#            metrics=[keras.metrics.binary_accuracy],
#        )
#
#        history = self.model.fit(
#            self.train_gen,
#            epochs=self.epochs,
#            verbose=self.verbose,
#            use_multiprocessing=False,
#            workers=4,
#            shuffle=True,
#        )
#
#    def get_embeddings(self):
#        x_inp_src = self.in_vec[0::2]
#        x_out_src = self.out_vec[0]
#        embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
#
#        node_ids = [*self.G.nodes()]
#        node_gen = GraphSAGENodeGenerator(
#            self.G, self.batch_size, self.num_samples
#        ).flow(node_ids)
#
#        node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=0)
#
#        return node_embeddings
