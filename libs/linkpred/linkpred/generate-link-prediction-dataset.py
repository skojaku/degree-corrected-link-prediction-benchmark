# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-01-17 03:57:05
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-01-17 09:35:10
# %%
import numpy as np
import pandas as pd
from scipy import sparse
import sys
from tqdm import tqdm
from node_samplers import SBMNodeSampler, ConfigModelNodeSampler, ErdosRenyiNodeSampler

if "snakemake" in sys.modules:
    train_net_file = snakemake.input["train_net_file"]
    test_net_file = snakemake.input["test_net_file"]
    node_table_file = snakemake.input["node_table_file"]
    samplerName = snakemake.params["samplerName"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/"
    output_file = "../data/"

# =================
# Load
# =================

train_net = sparse.load_npz(train_net_file)
test_net = sparse.load_npz(test_net_file)
node_table = pd.read_csv(node_table_file)

# ===============
# Preprocess
# ===============
net = train_net + test_net
group_cols = node_table["group_id"].values

# ===============
# Edge sampling
# ===============
def generate_positive_negative_edges(net, test_net, neg_edge_sampler):
    """Generate dataset for the link prediction.

    This function will generate the set of positive and negative edges
    by using negative edge sampler `neg_edge_sampler`.

    :param net: The adjacency matrix of the original network (before splitting test and train edges)
    :type net: scipy.sparse.csr_matrix
    :param test_net: The adjacency matrix of the test network (after the train-test split)
    :type test_net: scipy.sparse.csr_matrix
    :param neg_edge_sampler: edge node sampler
    :type neg_edge_sampler: see the node_sampler.py
    :return: pos_edges, neg_edges
    :rtype: pos_edges: tuple of node indices for positive edges, and neg_edges for the negative edges
    """

    #
    # Sampling positive edges
    #
    pos_src, pos_trg, _ = sparse.find(sparse.triu(test_net))
    n_test_edges = len(pos_src)
    n_nodes = net.shape[0]
    src, trg, _ = sparse.find(sparse.triu(net))

    # We represent a pair of integers by a complex number for computational ease.

    # Subscripts (integer pairs) to indices (a complex number)
    def sub2ind(r, c):
        return np.minimum(r, c) + 1j * np.maximum(r, c)

    # Inverse function of sub2ind
    def int2sub(z):
        return np.real(z).astype(int), np.imag(z).astype(int)

    # Represent the subscript pairs into complex numbers
    src_trg = sub2ind(src, trg)

    # prep. sampling the negative edges
    sampled_neg_edge_src_trg = set([])
    n_sampled = 0
    pbar = tqdm(total=n_test_edges)

    # Repeat until n_test_edges number of negative edges are sampled.
    while n_sampled < n_test_edges:

        # Sample negative edges based on SBM sampler
        _neg_src = np.random.choice(src, size=n_test_edges - n_sampled, replace=True)
        _neg_trg = neg_edge_sampler.sampling(_neg_src, _neg_src, n_nodes + 1)

        #
        # The sampled node pairs contain self loops, positive edges, and duplicates, which we remove here
        #
        # Remove self loops
        s = _neg_src != _neg_trg
        _neg_src, _neg_trg = _neg_src[s], _neg_trg[s]

        # To complex indices
        _neg_src_trg = sub2ind(_neg_src, _neg_trg)

        # Remove duplicates
        _neg_src_trg = np.unique(_neg_src_trg)

        # Remove positive edges
        s = ~np.isin(_neg_src_trg, src_trg)
        _neg_src_trg = _neg_src_trg[s]

        #
        # We add the survived negative edges to the list
        #
        sampled_neg_edge_src_trg.update(_neg_src_trg)

        # Update the progress bar
        diff = len(sampled_neg_edge_src_trg) - n_sampled
        n_sampled += diff
        pbar.update(diff)

    # To subscripts
    neg_src, neg_trg = int2sub(list(sampled_neg_edge_src_trg))

    # Make sure that no positive edge is included in the sampled negative edges.
    assert np.max(net[(neg_src, neg_trg)]) == 0
    pos_edges, neg_edges = (pos_src, pos_trg), (neg_src, neg_trg)
    return pos_edges, neg_edges


sampler = {
    "uniform": ErdosRenyiNodeSampler(),
    "degree-biased": ConfigModelNodeSampler(),
    "degree-group-biased": SBMNodeSampler(
        window_length=1, group_membership=group_cols, dcsbm=True
    ),
}[samplerName]

sampler = sampler.fit(net)

pos_edges, neg_edges = generate_positive_negative_edges(
    net,
    test_net,
    sampler,
)

# ===============
# Save
# ===============
pos_edges_src, pos_edges_trg = pos_edges
neg_edges_src, neg_edges_trg = neg_edges

isPositiveEdges, src, trg = (
    np.concatenate([np.ones_like(pos_edges_src), np.zeros_like(neg_edges_src)]),
    np.concatenate([pos_edges_src, neg_edges_src]),
    np.concatenate([pos_edges_trg, neg_edges_trg]),
)


pd.DataFrame({"src": src, "trg": trg, "isPositiveEdge": isPositiveEdges}).to_csv(
    output_file, index=False
)
