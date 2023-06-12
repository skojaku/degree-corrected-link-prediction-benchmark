# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-06-12 16:05:26
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-12 17:22:27
# %%
import unittest
import networkx as nx
from scipy import sparse
import numpy as np
from seal import gnns, node_samplers
from seal.seal import SEAL, train

# from seal.utils import train
import torch
import torch_geometric


class TestSEAL(unittest.TestCase):
    def test_seal_degree_debiased(self):
        G = nx.karate_club_graph()
        net = nx.adjacency_matrix(G)
        net = sparse.csr_matrix(net)
        net.eliminate_zeros()
        net.data = net.data * 0 + 1

        feature_vec = gnns.generate_base_embedding(net, 16)
        feature_vec = torch.FloatTensor(feature_vec)
        feature_dim = feature_vec.shape[1] + 1
        dim_h = 64
        dim = 64
        gnn_model = torch_geometric.nn.models.GCN(
            in_channels=feature_dim,
            hidden_channels=dim_h,
            num_layers=2,
            out_channels=dim,
        )

        model = SEAL(gnn_model=gnn_model)
        model = train(
            model=model,
            feature_vec=feature_vec,
            net=net,
            device="cuda:7",
            epochs=10,
            hops=2,
            feature_vec_dim=64,
            negative_edge_sampler=node_samplers.degreeBiasedNegativeEdgeSampling,
            # negative_edge_sampler=negative_uniform,
            batch_size=50,
            lr=0.01,
        )
        model.to("cpu")
        model.predict(1, 33, net=net)

    def test_seal(self):
        G = nx.karate_club_graph()
        net = nx.adjacency_matrix(G)
        net = sparse.csr_matrix(net)
        net.eliminate_zeros()
        net.data = net.data * 0 + 1

        feature_vec = gnns.generate_base_embedding(net, 16)
        feature_vec = torch.FloatTensor(feature_vec)
        feature_dim = feature_vec.shape[1] + 1
        dim_h = 64
        dim = 64
        gnn_model = torch_geometric.nn.models.GCN(
            in_channels=feature_dim,
            hidden_channels=dim_h,
            num_layers=2,
            out_channels=dim,
        )

        model = SEAL(gnn_model=gnn_model)
        model = train(
            model=model,
            feature_vec=feature_vec,
            net=net,
            device="cuda:7",
            epochs=10,
            hops=2,
            feature_vec_dim=64,
            negative_edge_sampler=node_samplers.negative_uniform,
            batch_size=50,
            lr=0.01,
        )
        model.to("cpu")
        model.predict(1, 33, net=net)


if __name__ == "__main__":
    unittest.main()
