# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-06-16 11:14:14
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-16 20:36:47
# %%
import torch
from collections import OrderedDict
import sys
import numpy as np


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

    def load(self, filename):
        raise NotImplementedError()
