# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-03-27 16:40:11
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-08-01 13:53:26
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from scipy.sparse import csgraph
from dclinkpred.node_samplers import (
    DegreeBiasedNodeSampler,
    UniformNodeSampler,
)

class LinkPredictionDataset:
    """
    Datasets for link prediction tasks.

    LinkPredictionDataset handles the splitting of a given network into training and testing sets,
    and prepares negative samples for training link prediction models.

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> net = csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> lp_dataset = LinkPredictionDataset(testEdgeFraction=0.2)
    >>> lp_dataset.fit(net)
    >>> train_net, target_edge_table = lp_dataset.transform()
    """

    def __init__(
        self,
        testEdgeFraction,
        degree_correction=True,
        negatives_per_positive=1,
        negative_edge_sampler_params={},
        allow_duplicated_negatives=False,
    ):
        """
        Initialize the LinkPredictionDataset with specified parameters.

        Parameters
        ----------
        testEdgeFraction : float
            Fraction of edges to be removed from the given network for testing.
        degree_correction : bool, optional
            If True, use degree correction in negative sampling. Default is True.
        negatives_per_positive : int, optional
            Number of negative edges to sample per positive edge. Defaults to 1.
        negative_edge_sampler_params : dict, optional
            Parameters for the negative edge sampler. Default is an empty dictionary.
        allow_duplicated_negatives : bool, optional
            If True, allows duplicated edges in negative samples. Default is False.
        """
        self.sampler = NegativeEdgeSampler(
            degree_correction=degree_correction,
            alllow_duplicated_negatives=allow_duplicated_negatives,
            **negative_edge_sampler_params
        )
        self.splitter = TrainTestEdgeSplitter(fraction=testEdgeFraction)
        self.testEdgeFraction = testEdgeFraction
        self.negatives_per_positive = negatives_per_positive
        self.allow_duplicated_negatives = allow_duplicated_negatives

    def fit(self, net):
        """
        Fit the LinkPredictionDataset model to the given network.

        This method prepares the dataset for link prediction by splitting the network into training and testing sets,
        and fitting the negative edge sampler to the training network.

        Parameters
        ----------
        net : scipy.sparse.csr_matrix, networkx.Graph, numpy.ndarray
            The adjacency matrix of the network.

        Returns
        -------
        self : LinkPredictionDataset
            The instance itself.
        """
        net, self.network_data_type = to_adjacency_matrix(net)
        self.n_nodes = net.shape[0]

        # Train-test edge split
        self.splitter.fit(net)

        train_src, train_trg = self.splitter.train_edges_

        # Ensure that the network is undirected and unweighted
        self.train_net = sparse.csr_matrix(
            (np.ones_like(train_src), (train_src, train_trg)),
            shape=(self.n_nodes, self.n_nodes),
        )
        self.train_net = sparse.csr_matrix(self.train_net + self.train_net.T)
        self.train_net.data = self.train_net.data * 0 + 1

        # Sampling negative edges
        self.sampler.fit(self.train_net)

        self.net = net
        return self

    def transform(self):
        """
        Transform the dataset for link prediction.

        This method generates the test and negative edges, and prepares the target edge table which includes both
        positive (test) and negative edges with labels.

        Returns
        -------
        tuple
            A tuple containing:
            - train_net : scipy.sparse.csr_matrix
                The training network as a sparse matrix.
            - src_test: np.ndarray
                Source nodes for test edges.
            - trg_test: np.ndarray
                Target nodes for test edges.
            - y_test: np.ndarray
                Binary labels for test edges, where 1 indicates a positive edge, and 0 indicates a negative edge.
        """

        test_src, test_trg = self.get_positive_edges()
        neg_src, neg_trg = self.get_negative_edges()
        src_test = np.concatenate([test_src, neg_src])
        trg_test = np.concatenate([test_trg, neg_trg])
        y_test = np.concatenate([np.ones_like(test_src), np.zeros_like(neg_trg)])

        # Shuffle the test edges
        order = np.random.permutation(len(src_test))
        src_test, trg_test, y_test = src_test[order], trg_test[order], y_test[order]

        _train_net = transform_network_data_type(self.train_net, to_type = self.network_data_type)

        return _train_net, src_test, trg_test, y_test

    def get_positive_edges(self):
        """
        Generate positive test edges for link prediction.

        This method retrieves the test edges that were previously split from the original graph.

        Returns
        -------
        tuple of np.ndarray
            A tuple containing two arrays:
            - First array contains the source nodes of the test edges.
            - Second array contains the target nodes of the test edges.
        """
        return self.splitter.test_edges_

    def get_negative_edges(self):
        """
        Generate negative test edges for link prediction.

        This method samples negative edges that are not present in the graph. The number of negative edges
        generated is determined by the `negatives_per_positive` attribute, which specifies how many negative
        edges should be generated per positive edge.

        Returns
        -------
        tuple of np.ndarray
            A tuple containing two arrays:
            - First array contains the source nodes of the negative edges.
            - Second array contains the target nodes of the negative edges.
        """


        test_src, test_trg = self.splitter.test_edges_
        n_test_edges = int(len(test_src))
        neg_src, neg_trg = [], []
        for _ in range(self.negatives_per_positive):
            _neg_src, _neg_trg = self.sampler.sampling(
                size=n_test_edges, test_edges=(test_src, test_trg)
            )
            neg_src.append(_neg_src)
            neg_trg.append(_neg_trg)
        neg_src, neg_trg = np.concatenate(neg_src), np.concatenate(neg_trg)
        return neg_src, neg_trg


class TrainTestEdgeSplitter:
    def __init__(self, fraction=0.25):
        """
        Initialize the TrainTestEdgeSplitter with a specified fraction of edges to be used as test edges.

        Parameters
        ----------
        fraction : float, optional
            The fraction of edges to be removed from the training graph and used as test edges. Default is 0.25.

        Examples
        --------
        >>> from scipy.sparse import csr_matrix
        >>> A = csr_matrix([
        ...     [0, 1, 0, 0],
        ...     [1, 0, 1, 0],
        ...     [0, 1, 0, 1],
        ...     [0, 0, 1, 0]
        ... ])
        >>> splitter = TrainTestEdgeSplitter(fraction=0.2)
        >>> splitter.fit(A)
        >>> test_edges = splitter.test_edges_
        >>> train_edges = splitter.train_edges_
        """
        self.fraction = fraction

    def __init__(self, fraction=0.25):
        """
        Initialize the TrainTestEdgeSplitter with a fraction of edges to be used as test edges.

        Parameters
        ----------
        fraction : float
            The fraction of edges to be removed from the graph and used as test edges.

        Attributes
        ----------
        fraction : float
            Stores the fraction of edges that will be used as test edges.
        """

        self.fraction = fraction

    def fit(self, A):
        """
        Fit the splitter to the adjacency matrix and prepare for edge splitting.

        Parameters
        ----------
        A : scipy.sparse.csr_matrix
            The adjacency matrix of the graph.

        Returns
        -------
        None
        """

        r, c, _ = sparse.find(A)
        edges = np.unique(pairing(r, c))

        MST = csgraph.minimum_spanning_tree(A)
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
        """
        Transform the graph by splitting it into training and testing edges.

        Returns
        -------
        tuple of np.ndarray
            A tuple containing two elements:
            - train_edges_: Array of training edges.
            - test_edges_: Array of testing edges.
        """

        return self.train_edges_, self.test_edges_


class NegativeEdgeSampler:
    def __init__(self, degree_correction=True, allow_duplicated_negatives=False, **params):
        """
        Initialize the NegativeEdgeSampler.

        Parameters
        ----------
        degree_correction : bool, optional
            If True, use degree-biased sampling, otherwise use uniform sampling.
            Default is True.
        allow_duplicated_negatives : bool, optional
            If True, allow duplicated negative edges in the sampling.
            Default is False.
        **params : dict
            Additional parameters for the sampler.

        Examples
        --------
        >>> from scipy.sparse import csr_matrix
        >>> net = csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        >>> sampler = NegativeEdgeSampler(degree_correction=True, allow_duplicated_negatives=False)
        >>> sampler.fit(net)
        >>> print(sampler.sampling(size=2))
        (array([1, 0]), array([2, 1]))
        """
        self.degree_correction = degree_correction
        self.allow_duplicated_negatives = allow_duplicated_negatives
        self.params = params

    def __init__(
        self, degree_correction=True, alllow_duplicated_negatives=False, **params
    ):
        """
        Initialize the NegativeEdgeSampler.

        Parameters
        ----------
        degree_correction : bool, optional
            If True, use degree-biased sampling, otherwise use uniform sampling.
            Default is True.
        alllow_duplicated_negatives : bool, optional
            If True, allow duplicated negative edges in the sampling.
            Default is False.
        **params : dict
            Additional parameters for future extensions or adjustments.
        """

        self.degree_correction = degree_correction
        self.allow_duplicated_negatives = alllow_duplicated_negatives

    def fit(self, net):
        """
        Fit the negative edge sampler to the network.

        Parameters
        ----------
        net : scipy.sparse.csr_matrix
            The adjacency matrix of the network.

        """

        if self.degree_correction:
            self.sampler = DegreeBiasedNodeSampler()
        else:
            self.sampler = UniformNodeSampler()

        self.net = net
        self.n_nodes = net.shape[0]
        src, trg, _ = sparse.find(sparse.triu(net))
        self.edge_indices = pairing(src, trg)
        self.sampler.fit(net)

    def sampling(self, size=None, test_edges=None):
        """
        Sample negative edges for the given network.

        Parameters
        ----------
        size : int, optional
            The number of negative edges to sample. If not specified, defaults to the number of test edges.
        test_edges : array_like, optional
            Array of test edges that should not be included in the negative samples.

        Returns
        -------
        sampled_neg_edge_indices : list
            List of sampled negative edge indices that do not conflict with the positive test edges and are not self-loops.
        """

        source_nodes = self.sampler.sampling(size=size, edge_sampling=False)

        sampled_neg_edge_indices = []
        n_sampled = 0

        # Repeat until n_test_edges number of negative edges are sampled.
        n_iters = 0
        max_iters = 30
        if test_edges is not None:
            test_edges = pairing(*test_edges)

        while (n_sampled < size) and (n_iters < max_iters):
            # Sample negative edges based on SBM sampler
            _neg_src, _neg_trg = self.sampler.sampling(size = len(source_nodes))

            # To edge indices for computation ease
            _neg_edge_indices = pairing(_neg_src, _neg_trg)

            #
            # The sampled node pairs contain self loops, positive edges, and duplicates, which we remove here
            #
            reject = np.full(len(_neg_src), False)

            # Remove _neg_edge_indices duplicated in self.edge_indices
            positivePairs = np.isin(_neg_edge_indices, self.edge_indices)
            reject[positivePairs] = True

            # Remove test edges from negative edges
            if test_edges is not None:
                positivePairs = np.isin(_neg_edge_indices, test_edges)
                reject[positivePairs] = True

            # Keep non-self-loops
            reject[_neg_src == _neg_trg] = True

            # Keep only the unique pairs
            if self.allow_duplicated_negatives == False:
                isUnique = np.full(len(_neg_src), False)
                isUnique[np.unique(_neg_edge_indices, return_index=True)[1]] = True
                reject[~isUnique] = True

                # Keep the pairs that have not been sampled
                existingPairs = np.isin(_neg_edge_indices, sampled_neg_edge_indices)
                reject[existingPairs] = True
            #
            # Add the survived negative edges to the list
            #
            sampled_neg_edge_indices += _neg_edge_indices[~reject].tolist()

            # Keep the rejected source nodes for resampling
            source_nodes = source_nodes[reject]

            # Update the progress bar
            diff = len(sampled_neg_edge_indices) - n_sampled
            n_sampled += diff

            n_iters += 1

        neg_src, neg_trg = depairing(np.array(sampled_neg_edge_indices))
        if len(neg_src) < size:
            ids = np.random.choice(len(neg_src), size=size - len(neg_src), replace=True)
            neg_src = np.concatenate([neg_src, neg_src[ids]])
            neg_trg = np.concatenate([neg_trg, neg_trg[ids]])
        return neg_src, neg_trg


def pairing(r, c):
    """
    Pair two arrays of row and column indices into a single array of complex numbers.

    Parameters
    ----------
    r : np.ndarray
        Array of row indices.
    c : np.ndarray
        Array of column indices.

    Returns
    -------
    np.ndarray
        Array of complex numbers where each complex number represents a pair (r, c).
    """
    return np.minimum(r, c) + 1j * np.maximum(r, c)


def depairing(v):
    """
    Decompose a complex number array into its real and imaginary parts as integers.

    Parameters
    ----------
    v : np.ndarray
        Array of complex numbers.

    Returns
    -------
    tuple
        A tuple containing two np.ndarray:
        - The first array contains the real parts of the complex numbers.
        - The second array contains the imaginary parts of the complex numbers.
    """

    return np.real(v).astype(int), np.imag(v).astype(int)


def to_adjacency_matrix(net):
    """
    Convert a given network into an adjacency matrix.

    Parameters
    ----------
    net : various types
        The network data which can be a scipy sparse matrix, networkx graph, or numpy ndarray.

    Returns
    -------
    tuple
        A tuple containing the adjacency matrix and a string indicating the type of the input network.
    """
    if sparse.issparse(net):
        if type(net) == "scipy.sparse.csr.csr_matrix":
            return net, "scipy.sparse"
        return sparse.csr_matrix(net), "scipy.sparse"
    elif "networkx" in "%s" % type(net):
        return nx.adjacency_matrix(net), "networkx"
    elif "numpy.ndarray" == type(net):
        return sparse.csr_matrix(net), "numpy.ndarray"

def transform_network_data_type(net, to_type, from_type = "scipy.sparse"):
    """
    Convert the network data type.

    Parameters
    ----------
    net : various types
        The network data which might be in different formats like scipy sparse matrix, networkx graph, or numpy array.
    to_type : str
        The target type to which the network data should be converted. Options are 'scipy.sparse', 'networkx', 'numpy.ndarray'.
    from_type : str, optional
        The original type of the network data. Default is 'scipy.sparse'.

    Returns
    -------
    converted_net : various types
        The network data converted to the specified type.
    """

    if from_type != "scipy.sparse":
        _net, _ = to_adjacency_matrix(net)
    else:
        _net = net

    if to_type == "scipy.sparse":
        return _net
    elif to_type == "networkx":
        return nx.from_scipy_sparse_array(_net)
    elif to_type == "numpy.ndarray":
        return _net.toarray()

