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
        """Fit the sampler.
        :param A: adjacency matrix
        :type A: scipy.csr_matrix
        :raises NotImplementedError: [description]
        """
        raise NotImplementedError

    def sampling(self, center_nodes, context_nodes, padding_id):
        """Sample context nodes from the graph for center nodes.
        :param center_nodes: ID of center node
        :type center_nodes: int
        :param context_nodes: ID of context node
        :type context_nodes: int
        :param n_samples: number of samples per center node
        :type n_samples: int
        :param padding_idx: ID of padding
        :type padding_id: int
        :raises NotImplementedError: [description]
        """
        raise NotImplementedError


class SBMNodeSampler(NodeSampler):
    """Node Sampler based on the stochatic block model."""

    def __init__(
        self,
        window_length=10,
        group_membership=None,
        dcsbm=True,
    ):
        """Node Sampler based on the stochastic block model.
        :param window_length: length of the context window, defaults to 10
        :type window_length: int, optional
        :param group_membership: group membership of nodes, defaults to None
        :type group_membership: np.ndarray, optional
        :param dcsbm: Set dcsbm=True to take into account the degree of nodes, defaults to True
        :type dcsbm: bool, optional
        """
        if group_membership is None:
            self.group_membership = None
        else:
            self.group_membership = np.unique(group_membership, return_inverse=True)[
                1
            ]  # reindex
        self.window_length = window_length
        self.dcsbm = dcsbm

    def fit(self, A):
        """Initialize the dcSBM sampler."""

        # Initalize the parameters
        self.n_nodes = A.shape[0]

        # Initialize the group membership
        if self.group_membership is None:
            self.group_membership = np.zeros(self.n_nodes, dtype=np.int64)
            self.node2group = to_member_matrix(self.group_membership)
        else:
            self.node2group = to_member_matrix(self.group_membership)

        indeg = np.array(A.sum(axis=0)).reshape(-1)
        Lambda = (self.node2group.T @ A @ self.node2group).toarray()
        Din = np.array(np.sum(Lambda, axis=0)).reshape(-1)
        Nin = np.array(self.node2group.sum(axis=0)).reshape(-1)
        Psbm = np.einsum(
            "ij,i->ij", Lambda, 1 / np.maximum(1, np.array(np.sum(Lambda, axis=1)))
        )
        Psbm_pow = matrix_sum_power(Psbm, self.window_length) / self.window_length

        if self.dcsbm:
            self.block2node = (
                sparse.diags(1 / np.maximum(1, Din))
                @ sparse.csr_matrix(self.node2group.T)
                @ sparse.diags(indeg)
            )
        else:
            self.block2node = sparse.diags(1 / np.maximum(1, Nin)) @ sparse.csr_matrix(
                self.node2group.T
            )

        # From block to block
        self.block2block = sparse.csr_matrix(Psbm_pow)
        self.block2block.data = _csr_row_cumsum(
            self.block2block.indptr, self.block2block.data
        )

        # From block to node
        self.block2node.data = _csr_row_cumsum(
            self.block2node.indptr, self.block2node.data
        )

        if self.dcsbm:
            self.p0 = np.maximum(indeg, 1) / np.sum(np.maximum(indeg, 1))
        else:
            self.p0 = np.ones_like(indeg, dtype="float32") / len(indeg)
        return self

    def sampling(self, size=None, center_nodes=None):
        if center_nodes is None:
            center_nodes = np.random.choice(
                self.n_nodes, size=size, p=self.p0, replace=True
            )
        block_ids = csr_sampling(self.group_membership[center_nodes], self.block2block)
        context_nodes = csr_sampling(block_ids, self.block2node)
        return center_nodes.astype(np.int64), context_nodes.astype(np.int64)

    def sampling_source_nodes(self, size):
        return np.random.choice(self.n_nodes, size=size, p=self.p0, replace=True)


class ConfigModelNodeSampler(SBMNodeSampler):
    def __init__(self):
        super(ConfigModelNodeSampler, self).__init__(window_length=1, dcsbm=True)


class ErdosRenyiNodeSampler(SBMNodeSampler):
    def __init__(self):
        super(ErdosRenyiNodeSampler, self).__init__(window_length=1, dcsbm=False)


#
# Homogenize the data format
#
def to_adjacency_matrix(net):
    """Convert to the adjacency matrix in form of sparse.csr_matrix.
    :param net: adjacency matrix
    :type net: np.ndarray or csr_matrix
    :return: adjacency matrix
    :rtype: sparse.csr_matrix
    """
    if sparse.issparse(net):
        if type(net) == "scipy.sparse.csr.csr_matrix":
            return net
        return sparse.csr_matrix(net)
    elif "numpy.ndarray" == type(net):
        return sparse.csr_matrix(net)
    else:
        ValueError("Unexpected data type {} for the adjacency matrix".format(type(net)))


def row_normalize(mat):
    """Normalize the matrix row-wise.
    :param mat: matrix
    :type mat: sparse.csr_matrix
    :return: row normalized matrix
    :rtype: sparse.csr_matrix
    """
    denom = np.array(mat.sum(axis=1)).reshape(-1).astype(float)
    return sparse.diags(1.0 / np.maximum(denom, 1e-32), format="csr") @ mat


def to_member_matrix(group_ids, node_ids=None, shape=None):
    """Create the binary member matrix U such that U[i,k] = 1 if i belongs to group k otherwise U[i,k]=0.
    :param group_ids: group membership of nodes. group_ids[i] indicates the ID (integer) of the group to which i belongs.
    :type group_ids: np.ndarray
    :param node_ids: IDs of the node. If not given, the node IDs are the index of `group_ids`, defaults to None.
    :type node_ids: np.ndarray, optional
    :param shape: Shape of the member matrix. If not given, (len(group_ids), max(group_ids) + 1), defaults to None
    :type shape: tuple, optional
    :return: Membership matrix
    :rtype: sparse.csr_matrix
    """
    if node_ids is None:
        node_ids = np.arange(len(group_ids))

    if shape is not None:
        Nr = int(np.max(node_ids) + 1)
        Nc = int(np.max(group_ids) + 1)
        shape = (Nr, Nc)
    U = sparse.csr_matrix(
        (np.ones_like(group_ids), (node_ids, group_ids)),
        shape=shape,
    )
    U.data = U.data * 0 + 1
    return U


def matrix_sum_power(A, T):
    """Take the sum of the powers of a matrix, i.e.,
    sum_{t=1} ^T A^t.
    :param A: Matrix to be powered
    :type A: np.ndarray
    :param T: Maximum order for the matrixpower
    :type T: int
    :return: Powered matrix
    :rtype: np.ndarray
    """
    At = np.eye(A.shape[0])
    As = np.zeros((A.shape[0], A.shape[0]))
    for _ in range(T):
        At = A @ At
        As += At
    return As


def pairing(k1, k2, unordered=False):
    """Cantor pairing function
    http://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function."""
    k12 = k1 + k2
    if unordered:
        return (k12 * (k12 + 1)) * 0.5 + np.minimum(k1, k2)
    else:
        return (k12 * (k12 + 1)) * 0.5 + k2


def depairing(z):
    """Inverse of Cantor pairing function http://en.wikipedia.org/wiki/Pairing_
    function#Inverting_the_Cantor_pairing_function."""
    w = np.floor((np.sqrt(8 * z + 1) - 1) * 0.5)
    t = (w**2 + w) * 0.5
    y = np.round(z - t).astype(np.int64)
    x = np.round(w - y).astype(np.int64)
    return x, y


def safe_log(A, minval=1e-12):
    if sparse.issparse(A):
        A.data = np.log(np.maximum(A.data, minval))
        return A
    else:
        return np.log(np.maximum(A, minval))


@njit(nogil=True)
def _csr_row_cumsum(indptr, data):
    out = np.empty_like(data)
    for i in range(len(indptr) - 1):
        acc = 0
        for j in range(indptr[i], indptr[i + 1]):
            acc += data[j]
            out[j] = acc
        out[j] = 1.0
    return out


def csr_sampling(rows, csr_mat):
    return _csr_sampling(rows, csr_mat.indptr, csr_mat.indices, csr_mat.data)


@njit(nogil=True)
def _neighbors(indptr, indices_or_data, t):
    return indices_or_data[indptr[t] : indptr[t + 1]]


@njit(nogil=True)
def _csr_sampling(rows, indptr, indices, data):
    n = len(rows)
    retval = np.empty(n, dtype=indices.dtype)
    for j in range(n):
        neighbors = _neighbors(indptr, indices, rows[j])
        neighbors_p = _neighbors(indptr, data, rows[j])
        retval[j] = neighbors[np.searchsorted(neighbors_p, np.random.rand())]
    return retval
