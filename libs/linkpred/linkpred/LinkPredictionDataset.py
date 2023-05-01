# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-03-27 16:40:11
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-04-22 22:05:57
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
from scipy.sparse import csgraph
from linkpred.node_samplers import ConfigModelNodeSampler, ErdosRenyiNodeSampler


class LinkPredictionDataset:
    """
    Generates a link prediction dataset for evaluating link prediction models.

    :param testEdgeFraction: Fraction of edges to be removed from the given network for testing.
    :type testEdgeFraction: float
    :param negative_edge_sampler: Type of negative edge sampler. Can be "uniform" for conventional link prediction evaluation or "degreeBiased" for degree-biased sampling.
    :type negative_edge_sampler: str
    :param negatives_per_positive: Number of negative edges to sample per positive edge. Defaults to 1.
    :type negatives_per_positive: int, optional
    :param conditionedOnSource: Whether to condition negative edge sampling on the source node of the positive edge. Defaults to False.
    :type conditionedOnSource: bool, optional

    Example usage:
    >> model = LinkPredictionDataset(testEdgeFraction=0.5, negative_edge_sampler="degreeBiased")
    >> model.fit(net)
    >> train_net, target_edge_table = model.transform()
    """

    def __init__(
        self,
        testEdgeFraction,
        negative_edge_sampler,
        negatives_per_positive=1,
        conditionedOnSource=False,
        all_negatives=False,
    ):
        """
        Initializer

        :param testEdgeFraction: Fraction of edges to be removed from the given network for testing.
        :type testEdgeFraction: float
        :param negative_edge_sampler: Type of negative edge sampler. Can be "uniform" for conventional link prediction evaluation or "degreeBiased" for degree-biased sampling.
        :type negative_edge_sampler: str
        :param negatives_per_positive: Number of negative edges to sample per positive edge. Defaults to 1.
        :type negatives_per_positive: int, optional
        :param conditionedOnSource: Whether to condition negative edge sampling on the source node of the positive edge. Defaults to False.
        :type conditionedOnSource: bool, optional
        """
        self.sampler = NegativeEdgeSampler(negative_edge_sampler=negative_edge_sampler)
        self.splitter = TrainTestEdgeSplitter(fraction=testEdgeFraction)
        self.testEdgeFraction = testEdgeFraction
        self.negatives_per_positive = negatives_per_positive
        self.conditionedOnSource = conditionedOnSource
        self.all_negatives = all_negatives

    def fit(self, net):
        self.n_nodes = net.shape[0]

        # Train-test edge split
        self.splitter.fit(net)

        # Sampling negative edges
        self.sampler.fit(net)

        self.net = net

    def transform(self):
        test_src, test_trg = self.splitter.test_edges_
        train_src, train_trg = self.splitter.train_edges_

        # Ensure that the network is undirected and unweighted
        self.train_net = sparse.csr_matrix(
            (np.ones_like(train_src), (train_src, train_trg)),
            shape=(self.n_nodes, self.n_nodes),
        )
        self.train_net = sparse.csr_matrix(self.train_net + self.train_net.T)
        self.train_net.data = self.train_net.data * 0 + 1

        if self.all_negatives:
            # We evaluate the all positives and all negatives
            neg_src, neg_trg = np.triu_indices(self.n_nodes, k=1)
            y = np.array(self.net[(neg_src, neg_trg)]).reshape(-1)
            s = y == 0
            neg_src, neg_trg, y = neg_src[s], neg_trg[s], y[s]

            self.target_edge_table = pd.DataFrame(
                {
                    "src": np.concatenate([test_src, neg_src]),
                    "trg": np.concatenate([test_trg, neg_trg]),
                    "isPositiveEdge": np.concatenate(
                        [np.ones_like(test_src), np.zeros_like(neg_trg)]
                    ),
                }
            )
            return self.train_net, self.target_edge_table

        n_test_edges = np.int(len(test_src))
        neg_src, neg_trg = [], []
        for _ in range(self.negatives_per_positive):
            if self.conditionedOnSource:
                _neg_src, _neg_trg = self.sampler.sampling(source_nodes=test_src)
            else:
                _neg_src, _neg_trg = self.sampler.sampling(size=n_test_edges)
            neg_src.append(_neg_src)
            neg_trg.append(_neg_trg)
        neg_src, neg_trg = np.concatenate(neg_src), np.concatenate(neg_trg)

        self.target_edge_table = pd.DataFrame(
            {
                "src": np.concatenate([test_src, neg_src]),
                "trg": np.concatenate([test_trg, neg_trg]),
                "isPositiveEdge": np.concatenate(
                    [np.ones_like(test_src), np.zeros_like(neg_trg)]
                ),
            }
        )
        return self.train_net, self.target_edge_table


class TrainTestEdgeSplitter:
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


class NegativeEdgeSampler:
    def __init__(self, negative_edge_sampler):
        self.sampler = {
            "uniform": ErdosRenyiNodeSampler(),
            "degreeBiased": ConfigModelNodeSampler(),
        }[negative_edge_sampler]

    def fit(self, net):
        self.net = net
        self.n_nodes = net.shape[0]
        src, trg, _ = sparse.find(sparse.triu(net))
        self.edge_indices = pairing(src, trg)
        self.sampler.fit(net)

    def sampling(self, size=None, source_nodes=None):
        """
        Generates a dataset for link prediction by sampling positive and negative edges using the specified negative edge sampler.

        :param size: Number of edges to sample. Defaults to None.
        :type size: int, optional
        :param source_nodes: List of source nodes to condition negative edge sampling on. Defaults to None.
        :type source_nodes: list, optional
        :return: Tuple of node indices for positive edges (pos_edges) and negative edges (neg_edges)
        :rtype: tuple
        """
        # prep. sampling the negative edges
        if source_nodes is None:
            source_nodes = self.sampler.sampling_source_nodes(size=size)
        else:
            size = len(source_nodes)

        sampled_neg_edge_indices = set([])
        n_sampled = 0
        pbar = tqdm(total=size)

        # Repeat until n_test_edges number of negative edges are sampled.
        n_iters = 0
        max_iters = 30
        while (n_sampled < size) and (n_iters < max_iters):
            # Sample negative edges based on SBM sampler
            _neg_src, _neg_trg = self.sampler.sampling(center_nodes=source_nodes)

            # To edge indices for computation ease
            _neg_edge_indices = pairing(_neg_src, _neg_trg)
            #
            # The sampled node pairs contain self loops, positive edges, and duplicates, which we remove here
            #
            reject = np.full(len(_neg_src), False)

            # Keep non-self-loops
            reject[_neg_src == _neg_trg] = True

            # Keep only the unique pairs
            isUnique = np.full(len(_neg_src), False)
            isUnique[np.unique(_neg_edge_indices, return_index=True)[1]] = True
            reject[~isUnique] = True

            # Keep the pairs that have not been sampled
            existingPairs = np.isin(_neg_edge_indices, sampled_neg_edge_indices)
            reject[existingPairs] = True

            #
            # Add the survived negative edges to the list
            #
            sampled_neg_edge_indices.update(_neg_edge_indices[~reject])

            # Keep the rejected source nodes for resampling
            source_nodes = source_nodes[reject]

            # Update the progress bar
            diff = len(sampled_neg_edge_indices) - n_sampled
            n_sampled += diff
            pbar.update(diff)

            n_iters += 1

        neg_src, neg_trg = depairing(np.array(list(sampled_neg_edge_indices)))
        if len(neg_src) < size:
            ids = np.random.choice(len(neg_src), size=size - len(neg_src), replace=True)
            neg_src = np.concatenate([neg_src, neg_src[ids]])
            neg_trg = np.concatenate([neg_trg, neg_trg[ids]])

        return neg_src, neg_trg


def pairing(r, c):
    return np.minimum(r, c) + 1j * np.maximum(r, c)


def depairing(v):
    return np.real(v).astype(int), np.imag(v).astype(int)
