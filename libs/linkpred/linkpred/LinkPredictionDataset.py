# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-03-27 16:40:11
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-04-02 05:16:43
# %%
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
from scipy.sparse import csgraph
from linkpred.node_samplers import ConfigModelNodeSampler, ErdosRenyiNodeSampler


class LinkPredictionDataset:
    """Generate a link prediction dataset.

    >> model = LinkPredictionDataset(testEdgeFraction=0.5, negative_edge_sampler="degreeBiased")
    >> model.fit(net)
    >> train_net, target_edge_table = model.transform()

    """

    def __init__(self, testEdgeFraction, negative_edge_sampler):
        """Initializer

        :param testEdgeFraction: fraction of edges to be removed from the given network
        :type testEdgeFraction: float
        :param negative_edge_sampler: type of negative edge sampler. "uniform" for the conventional link prediction evaluation. "degreeBiased" for the degree-biased one.
        :type negative_edge_sampler: _type_
        """
        self.sampler = {
            "uniform": ErdosRenyiNodeSampler(),
            "degreeBiased": ConfigModelNodeSampler(),
        }[negative_edge_sampler]
        self.testEdgeFraction = testEdgeFraction

        self.splitter = NetworkTrainTestSplitterWithMST(fraction=testEdgeFraction)

    def fit(self, net):

        src, trg, _ = sparse.find(sparse.triu(net))
        n_nodes = net.shape[0]

        # Train-test edge split
        self.splitter.fit(net)
        test_src, test_trg = self.splitter.test_edges_
        train_src, train_trg = self.splitter.train_edges_

        # Sampling negative edges
        self.sampler.fit(net)
        pos_edges, neg_edges = self.generate_positive_negative_edges(
            src, trg, test_src, test_trg, n_nodes, self.sampler
        )

        self.train_net = sparse.csr_matrix(
            (np.ones_like(train_src), (train_src, train_trg)), shape=(n_nodes, n_nodes)
        )

        # Ensure that the network is undirected and unweighted
        self.train_net = sparse.csr_matrix(self.train_net + self.train_net.T)
        self.train_net.data = self.train_net.data * 0 + 1

        self.target_edge_table = pd.DataFrame(
            {
                "src": np.concatenate([pos_edges[0], neg_edges[0]]),
                "trg": np.concatenate([pos_edges[1], neg_edges[1]]),
                "isPositiveEdge": np.concatenate(
                    [np.ones_like(pos_edges[1]), np.zeros_like(neg_edges[1])]
                ),
            }
        )

    def transform(self):
        return self.train_net, self.target_edge_table

    def generate_positive_negative_edges(
        self, src, trg, test_src, test_trg, n_nodes, neg_edge_sampler
    ):
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
        n_test_edges = len(test_src)

        # We represent a pair of integers by a complex number for computational ease.

        # Represent the subscript pairs into complex numbers
        src_trg = pairing(src, trg)

        # prep. sampling the negative edges
        sampled_neg_edge_src_trg = set([])
        n_sampled = 0
        pbar = tqdm(total=n_test_edges)

        # Repeat until n_test_edges number of negative edges are sampled.
        while n_sampled < n_test_edges:

            # Sample negative edges based on SBM sampler
            _neg_src, _neg_trg = neg_edge_sampler.sampling(n_test_edges - n_sampled)

            #
            # The sampled node pairs contain self loops, positive edges, and duplicates, which we remove here
            #
            # Remove self loops
            s = _neg_src != _neg_trg
            _neg_src, _neg_trg = _neg_src[s], _neg_trg[s]

            # To complex indices
            _neg_src_trg = pairing(_neg_src, _neg_trg)

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
        neg_src, neg_trg = depairing(np.array(list(sampled_neg_edge_src_trg)))

        # Make sure that no positive edge is included in the sampled negative edges.
        pos_edges, neg_edges = (test_src, test_trg), (neg_src, neg_trg)
        return pos_edges, neg_edges


class NetworkTrainTestSplitterWithMST:
    def __init__(self, fraction=0.5):
        """Only support undirected Network.

        :param G: Networkx graph object. Origin Graph
        :param fraction: Fraction of edges that will be removed (test_edge).
        """
        self.fraction = fraction

    def fit(self, A):
        """Split train and test edges with MST.

        Train network should have a one weakly connected component.
        """
        r, c, _ = sparse.find(A)
        edges = np.unique(pairing(r, c))

        MST = csgraph.minimum_spanning_tree(A + A.T)
        r, c, _ = sparse.find(MST)
        mst_edges = np.unique(pairing(r, c))
        remained_edge_set = np.array(
            list(set(list(edges)).difference(set(list(mst_edges))))
        )
        n_edge_removal = int(len(edges) * self.fraction)
        if len(remained_edge_set) < n_edge_removal:
            raise Exception(
                "Cannot remove edges by keeping the connectedness. Decrease the `fraction` parameter"
            )

        test_edge_set = np.random.choice(
            remained_edge_set, n_edge_removal, replace=False
        )

        train_edge_set = np.array(
            list(set(list(edges)).difference(set(list(test_edge_set))))
        )

        self.test_edges_ = depairing(test_edge_set)
        self.train_edges_ = depairing(train_edge_set)
        self.n = A.shape[0]

    def transform(self):
        return self.train_edges_, self.test_edges_


def pairing(r, c):
    return np.minimum(r, c) + 1j * np.maximum(r, c)


def depairing(v):
    return np.real(v).astype(int), np.imag(v).astype(int)


# %%
