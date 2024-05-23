# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-04-11 16:47:31
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-04-11 16:51:24
from scipy import sparse
import numpy as np


def to_undirected(net):
    net = sparse.csr_matrix(net + net.T)
    net.data = net.data * 0 + 1
    net.eliminate_zeros()
    return net


def edgeList2adjacencyMatrix(src, trg, n_nodes):
    net = sparse.csr_matrix((np.ones_like(src), (src, trg)), shape=(n_nodes, n_nodes))
    return to_undirected(net)


def pairing(r, c):
    return np.minimum(r, c) + 1j * np.maximum(r, c)


def depairing(v):
    return np.real(v).astype(int), np.imag(v).astype(int)
