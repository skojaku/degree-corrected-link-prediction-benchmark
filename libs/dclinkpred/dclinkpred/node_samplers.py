# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-01-16 17:34:35
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-12 15:41:22

"""Graph module to store a network and generate random walks from it."""
import numpy as np
from scipy import sparse
from numba import njit


class NodeSampler:
    def fit(self, A):
        """
        Fit the node sampler with the adjacency matrix of the graph.

        Parameters
        ----------
        A : scipy.sparse.csr_matrix
            The adjacency matrix of the graph.

        Raises
        ------
        NotImplementedError
            If this method is not implemented in the subclass.
        """
        raise NotImplementedError

    def sampling(self, size=None, src_nodes=None, edge_sampling = True):
        """
        Sample nodes or node pairs from the graph.

        Parameters
        ----------
        size : int, optional
            The number of nodes or node pairs to sample. If not specified, a single node or node pair is sampled.
        src_nodes : array_like, optional
            Array of source node indices. If specified, sampling will be restricted to these nodes.
        edge_sampling : bool, optional
            If True, returns a tuple of arrays (src_nodes, trg_nodes). If False, returns only src_nodes.

        Returns
        -------
        np.ndarray or tuple of np.ndarray
            If edge_sampling is False, returns an array of sampled node indices.
            If edge_sampling is True, returns a tuple (src_nodes, trg_nodes) where both are arrays of sampled node indices.
        """
        raise NotImplementedError


class DegreeBiasedNodeSampler(NodeSampler):
    """
    This sampler biases the node sampling process towards nodes with higher degrees.

    Example
    -------
    >>> from scipy.sparse import csr_matrix
    >>> A = csr_matrix([
    ...     [0, 1, 0, 0],
    ...     [1, 0, 1, 0],
    ...     [0, 1, 0, 1],
    ...     [0, 0, 1, 0]
    ... ])
    >>> sampler = DegreeBiasedNodeSampler()
    >>> sampler.fit(A)
    >>> print(sampler.sampling(size=2, edge_sampling=False))
    [1 2]
    >>> print(sampler.sampling(size=2, edge_sampling=True))
    [1 2], [1,3]
    """
    def __init__(self):
       pass

    def fit(self, A):
        """
        Fit the DegreeBiasedNodeSampler with the adjacency matrix of the graph.

        This method calculates the degree of each node and sets up a probability distribution
        for sampling based on node degrees. The probability of sampling a node is proportional
        to its degree.

        Parameters
        ----------
        A : scipy.sparse.csr_matrix
            The adjacency matrix of the graph.
        """

        self.A = A
        self.deg = np.array(A.sum(axis=1)).reshape(-1)
        self.p = self.deg / self.deg.sum()
        self.n_nodes = A.shape[0]

    def sampling(self, size, src_nodes=None, edge_sampling = True):
        """
        Sample nodes with a degree-biased probability from the graph.

        Parameters
        ----------
        size : int
            The number of nodes or node pairs to sample.
        src_nodes : array_like, optional
            Array of source node indices. If specified, sampling will be restricted to these nodes.
        edge_sampling : bool, optional
            If True, returns a tuple of arrays (src_nodes, trg_nodes). If False, returns only src_nodes.

        Returns
        -------
        np.ndarray or tuple of np.ndarray
            If edge_sampling is False, returns an array of sampled node indices.
            If edge_sampling is True, returns a tuple (src_nodes, trg_nodes) where both are arrays of sampled node indices.
        """

        if src_nodes is None:
            src_nodes = np.random.choice(
                self.n_nodes, size=size, p=self.p, replace=True
            )

        if edge_sampling is False:
            return src_nodes.astype(np.int64)

        trg_nodes = np.random.choice(self.n_nodes, size, p=self.p, replace=True)
        return src_nodes.astype(np.int64), trg_nodes.astype(np.int64)


class UniformNodeSampler(NodeSampler):
    """
    This sampler samples nodes uniformly from the graph

    Example
    -------
    >>> from scipy.sparse import csr_matrix
    >>> A = csr_matrix([
    ...     [0, 1, 0, 0],
    ...     [1, 0, 1, 0],
    ...     [0, 1, 0, 1],
    ...     [0, 0, 1, 0]
    ... ])
    >>> sampler = UniformNodeSampler()
    >>> sampler.fit(A)
    >>> print(sampler.sampling(size=2, edge_sampling=False))
    [1 2]
    >>> print(sampler.sampling(size=2, edge_sampling=True))
    [1 2], [1,3]
    """

    def __init__(self):
        """
        Initialize the UniformNodeSampler.
        """
        pass

    def fit(self, A):
        """
        Fit the UniformNodeSampler with the adjacency matrix of the graph.

        Parameters
        ----------
        A : scipy.sparse.csr_matrix
            The adjacency matrix of the graph.
        """
        self.A = A
        self.n_nodes = A.shape[0]

    def sampling(self, size=None, src_nodes=None, edge_sampling=True):
        """
        Sample nodes uniformly from the graph.

        Parameters
        ----------
        size : int, optional
            The number of nodes to sample. If not specified, all nodes are sampled.
        src_nodes : array_like, optional
            Array of source node indices. If specified, sampling will be restricted to these nodes.
        edge_sampling : bool, optional
            If True, returns a tuple of arrays (src_nodes, trg_nodes). If False, returns only src_nodes.

        Returns
        -------
        np.ndarray or tuple of np.ndarray
            If edge_sampling is False, returns an array of sampled node indices.
            If edge_sampling is True, returns a tuple (src_nodes, trg_nodes) where both are arrays of sampled node indices.

        """
        if src_nodes is None:
            src_nodes = np.random.choice(self.n_nodes, size=size, replace=True)
        if edge_sampling is False:
            return src_nodes.astype(np.int64)

        trg_nodes = np.random.choice(self.n_nodes, size, replace=True)
        return src_nodes.astype(np.int64), trg_nodes.astype(np.int64)

    def __init__(self):
        pass

    def fit(self, A):
        """
        Fit the UniformNodeSampler with the adjacency matrix of the graph.

        Parameters
        ----------
        A : scipy.sparse.csr_matrix
            The adjacency matrix of the graph.
        """

        self.A = A
        self.n_nodes = A.shape[0]

    def sampling(self, size=None, src_nodes=None, edge_sampling = True):
        """
        Sample nodes uniformly from the graph.

        Parameters
        ----------
        size : int, optional
            The number of nodes to sample. If not specified, all nodes are sampled.
        src_nodes : array_like, optional
            Array of source node indices. If specified, sampling will be restricted to these nodes.
        edge_sampling : bool, optional
            If True, pairs of nodes (edges) are sampled. If False, only source nodes are sampled.

        Returns
        -------
        np.ndarray
            If edge_sampling is False, returns an array of sampled source node indices.
            If edge_sampling is True, returns a tuple of arrays (src_indices, trg_indices),
            where `src_indices` are the source node indices and `trg_indices` are the target node indices.
        """

        if src_nodes is None:
            src_nodes = np.random.choice(
                self.n_nodes, size=size, replace=True
            )
        if edge_sampling is False:
            return src_nodes.astype(np.int64)

        trg_nodes = np.random.choice(self.n_nodes, size, replace=True)
        return src_nodes.astype(np.int64), trg_nodes.astype(np.int64)