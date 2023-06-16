# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-06-16 11:14:14
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-16 18:10:35
# %%
import torch
from EmbeddingModels import embedding_models
from NetworkTopologyPredictionModels import topology_models
from collections import OrderedDict
import sys
import numpy as np
from seal import gnns, node_samplers
from seal.seal import SEAL, train
from seal.node_samplers import negative_uniform, degreeBiasedNegativeEdgeSampling
import torch
import torch_geometric


class LinkPredictor(torch.nn.Module):
    def predict(network, src, trg):
        raise NotImplementedError()

    def train(network, **params):
        raise NotImplementedError()

    def state_dict(self):
        d = super().state_dict()
        d.update(self.params)
        d["model"] = self.model
        return d


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


class NetworkLinkPredictor(LinkPredictor):
    def __init__(self, model, **params):
        self.model = model
        self.params = params

    def train(self, network, **params):
        pass

    def predict(self, network, src, trg, **params):
        return topology_models[self.model](network=network, src=src, trg=trg)

    def state_dict(self):
        d = OrderedDict({"model": self.model})
        d.update(self.params)
        return d

    def load(self, filename):
        d = torch.load(filename)
        self.model = d["model"]
        self.params = d


class SEALLinkPredictor(LinkPredictor):
    def __init__(self, model, device="cpu", **params):
        super(SEALLinkPredictor, self).__init__()
        self.model = model
        self.params = params
        self.device = device

    def load_gnn(self, **params):
        return torch_geometric.nn.models.__dict__[params["gnn_model"]](**params)

    def train(self, network, **params):
        feature_vec = gnns.generate_base_embedding(
            network, self.params["model_params"]["in_channels"]
        )
        self.params["model_params"]["in_channels"] += 1

        gnn_model = self.load_gnn(**self.params["model_params"])

        # gnn_model = torch_geometric.nn.models.GCN(**self.params["model_params"])
        self.params["train_params"]["negative_edge_sampler"] = (
            node_samplers.degreeBiasedNegativeEdgeSampling
            if self.params["train_params"]["negative_edge_sampler"] == "degreeBiased"
            else negative_uniform
        )

        gnn_model = train(
            model=gnn_model,
            feature_vec=torch.FloatTensor(feature_vec),
            net=network,
            device=self.device,
            **self.params["train_params"]
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
        self.gnn_model = self.load_gnn(**d["model_params"])
        self.feature_vec = d["feature_vec"]
        self.model = d["model"]
        self.params = d
        self.seal_model = SEAL(gnn_model=self.gnn_model, feature_vec=self.feature_vec)
