# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-03-27 17:59:10
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-12 16:27:34
from collections import Counter

import networkx as nx
import numba
import numpy as np
from scipy import sparse


def sample_unconnected_node_pairs(
    edge_index, num_nodes, num_samples, sampler, max_it=10
):
    # Function to sample negative edges for link prediction tasks
    n_sampled = 0
    pos_edge_ids = set(
        list(pairing(edge_index[0], edge_index[1]))
    )  # Get positive edge indices from the given edge_index
    n_pos_edges = len(pos_edge_ids)  # Count number of positive edges
    sampled = pos_edge_ids.copy()  # Initialize a set with positive edge indices
    it = 0
    while (n_sampled < num_samples) and (it < max_it):
        _edge_index = sampler(
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_neg_samples=2,
        )  # Sample negative edge indices using the given sampler
        _edge_index = list(
            pairing(_edge_index[0], _edge_index[1])
        )  # Convert the sampled edge index format to a list of edge indices
        sampled.update(_edge_index)  # Add the sampled indices to the set
        n_sampled = (
            len(sampled) - n_pos_edges
        )  # Count the number of edges sampled so far
        it += 1

    neg_edge_ids = list(
        sampled.difference(pos_edge_ids)
    )  # Remove positive edges from the set to get negative edges
    neg_edge_ids = np.random.choice(
        neg_edge_ids, size=num_samples, replace=False
    )  # Randomly select target negative edges from the set
    neg_edge_index = depairing(
        neg_edge_ids, vstack=True
    )  # Convert the negative edge indices back to the original format and return
    return neg_edge_index


#
# Homogenize the data format
#
def to_adjacency_matrix(net):
    if sparse.issparse(net):
        if type(net) == "scipy.sparse.csr.csr_matrix":
            return net
        return sparse.csr_matrix(net)
    elif "networkx" in "%s" % type(net):
        return nx.adjacency_matrix(net)
    elif "numpy.ndarray" == type(net):
        return sparse.csr_matrix(net)


def toUndirected(net):
    net = net + net.T
    net.data = net.data * 0 + 1
    net.eliminate_zeros()
    net = sparse.csr_matrix.asfptype(net)
    return net


def adj2edgeindex(net):
    src, trg, _ = sparse.find(net)
    return np.vstack([src, trg])


def edge2network(src, trg, n_nodes=None, val=None):
    if val is None:
        val = np.ones_like(src)
    if n_nodes is None:
        n_nodes = np.max([np.max(src), np.max(trg)]) + 1
    return toUndirected(sparse.csr_matrix((val, (src, trg)), shape=(n_nodes, n_nodes)))


def pairing(r, c):
    return np.minimum(r, c) + 1j * np.maximum(r, c)


def depairing(v, vstack=False):
    row, col = np.real(v).astype(int), np.imag(v).astype(int)
    if vstack:
        return np.vstack([row, col])
    else:
        return row, col
