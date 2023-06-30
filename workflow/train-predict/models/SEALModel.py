# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-06-16 20:32:43
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-30 17:12:00
import networkx as nx
from scipy import sparse
import numpy as np
import networkx as nx
from seal import gnns, node_samplers
from seal.seal import SEAL, train
import torch
import torch_geometric
from .ModelTemplate import LinkPredictor


class SEALLinkPredictor(LinkPredictor):
    def __init__(self, model, device="cpu", **params):
        super(SEALLinkPredictor, self).__init__()
        self.model = model
        self.params = params
        self.device = device

    def load_gnn(self, **params):
        gnn_model = self.model.split("+")[-1]
        return torch_geometric.nn.models.__dict__[gnn_model](**params)

    def train(self, network, **params):
        feature_vec = gnns.generate_base_embedding(network, self.params["in_channels"])
        self.params["in_channels"] += 1

        gnn_model = self.load_gnn(**self.params)

        self.params["negative_edge_sampler"] = (
            node_samplers.degreeBiasedNegativeEdgeSampling
            if self.params["negative_edge_sampler"] == "degreeBiased"
            else node_samplers.negative_uniform
        )

        gnn_model = train(
            model=gnn_model,
            feature_vec=torch.FloatTensor(feature_vec),
            net=network,
            device=self.device,
            **self.params
        )
        gnn_model.to("cpu")
        self.seal_model = SEAL(gnn_model=gnn_model, feature_vec=feature_vec)
        self.gnn_model = gnn_model
        self.feature_vec = feature_vec

    def predict(self, network, src, trg, **params):
        return self.seal_model.predict(network, src, trg, **params)

    def state_dict(self):
        d = super().state_dict()
        d.update(self.params)
        d["model"] = self.model
        d["gnn_model"] = self.gnn_model.state_dict()
        d["feature_vec"] = torch.FloatTensor(self.feature_vec)
        return d

    def load(self, filename):
        d = torch.load(filename)
        self.gnn_model = self.load_gnn(**d)
        self.feature_vec = d["feature_vec"]
        self.model = d["model"]
        self.params = d
        self.seal_model = SEAL(gnn_model=self.gnn_model, feature_vec=self.feature_vec)