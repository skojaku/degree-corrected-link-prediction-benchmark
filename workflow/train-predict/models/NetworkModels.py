# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 15:08:01
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-30 16:46:50
from scipy import sparse
import numpy as np
from .ModelTemplate import LinkPredictor
from collections import OrderedDict
import torch


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


topology_models = {}
topology_model = lambda f: topology_models.setdefault(f.__name__, f)


@topology_model
def preferentialAttachment(network, src, trg):
    deg = np.array(network.sum(axis=1)).reshape(-1)
    return deg[src] * deg[trg]


@topology_model
def commonNeighbors(network, src, trg):
    return np.array((network[src, :].multiply(network[trg, :])).sum(axis=1)).reshape(-1)


@topology_model
def jaccardIndex(network, src, trg):
    deg = np.array(network.sum(axis=1)).reshape(-1)
    score = np.array((network[src, :].multiply(network[trg, :])).sum(axis=1)).reshape(
        -1
    )
    return score / np.maximum(deg[src] + deg[trg] - score, 1)


@topology_model
def resourceAllocation(network, src, trg):
    deg = np.array(network.sum(axis=1)).reshape(-1)
    deg_inv = 1 / np.maximum(deg, 1)
    deg_inv[deg == 0] = 0
    return np.array(
        ((network[src, :] @ sparse.diags(deg_inv)).multiply(network[trg, :])).sum(
            axis=1
        )
    ).reshape(-1)


@topology_model
def adamicAdar(network, src, trg):
    deg = np.array(network.sum(axis=1)).reshape(-1)
    log_deg_inv = 1 / np.maximum(np.log(np.maximum(deg, 1)), 1)
    log_deg_inv[deg == 0] = 0
    return np.array(
        ((network[src, :] @ sparse.diags(log_deg_inv)).multiply(network[trg, :])).sum(
            axis=1
        )
    ).reshape(-1)


@topology_model
def localRandomWalk(network, src, trg):
    deg = np.array(network.sum(axis=1)).reshape(-1)
    deg_inv = 1 / np.maximum(deg, 1)
    P = sparse.diags(deg_inv) @ network
    PP = P @ P
    PPP = PP @ P
    S = P + PP + PPP
    S = sparse.diags(deg / np.sum(deg)) @ S
    return np.array(S[(src, trg)]).reshape(-1)


@topology_model
def localPathIndex(network, src, trg, epsilon=1e-3):
    A = network
    AA = A @ A
    AAA = AA @ A
    S = AA + epsilon * AAA
    return np.array(S[(src, trg)]).reshape(-1)
