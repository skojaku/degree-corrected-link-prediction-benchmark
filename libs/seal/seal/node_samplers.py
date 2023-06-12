# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-06-12 16:08:27
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-12 16:08:46
import numpy as np
import torch


def negative_uniform(edge_index, num_nodes, num_neg_samples):
    t = np.random.randint(
        0, num_nodes, size=num_neg_samples * edge_index.size()[1]
    ).reshape((num_neg_samples, edge_index.size()[1]))
    return torch.LongTensor(t)


def degreeBiasedNegativeEdgeSampling(edge_index, num_nodes, num_neg_samples):
    deg = np.bincount(edge_index.reshape(-1).cpu(), minlength=num_nodes).astype(float)
    deg /= np.sum(deg)
    t = np.random.choice(
        num_nodes, p=deg, size=num_neg_samples * edge_index.size()[1]
    ).reshape((num_neg_samples, edge_index.size()[1]))
    return torch.LongTensor(t)
