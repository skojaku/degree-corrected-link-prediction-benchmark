# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-06-16 20:41:32
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-30 16:46:53
import torch
import networkx as nx
from scipy import sparse
import numpy as np
import networkx as nx
from .ModelTemplate import LinkPredictor
import stacklp


class StackingLinkPredictor(LinkPredictor):
    def __init__(self, model, **params):
        super(StackingLinkPredictor, self).__init__()
        self.model = model
        self.params = params
        self.pred_model = None

    def train(self, network, **params):
        self.pred_model = stacklp.StackingLinkPredictionModel(**self.params)
        self.pred_model.fit(network)

    def predict(self, network, src, trg, **params):
        return self.pred_model.predict(network, src, trg)

    def state_dict(self):
        d = super().state_dict()
        d.update(self.params)
        d["pred_model"] = self.pred_model
        return d

    def load(self, filename):
        d = torch.load(filename)
        self.pred_model = d["pred_model"]
        self.model = d["model"]
        self.params = d
