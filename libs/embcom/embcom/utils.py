import logging
import numbers
from collections import Counter

import networkx as nx
import numba
import numpy as np
from numba import prange
from scipy import sparse

logger = logging.getLogger(__name__)


#
# Compute the transition matrix
#
def calc_rwr(P, r, max_step, offset=0, w=None):
    if w is None:
        w = np.power(1 - r, np.arange(max_step))
        w = w / np.sum(w)

    Pt = sparse.csr_matrix(sparse.diags(np.ones(P.shape[0])))
    Ps = sparse.csr_matrix(sparse.diags(np.zeros(P.shape[0])))
    for i in range(max_step):
        Pt = P @ Pt
        if i < offset:
            continue
        Ps = Ps + w[i] * Pt
    return Ps


def calc_stationary_dist(P, itnum=30):
    N = P.shape[0]
    x = np.ones((1, N)) / N
    xt = np.ones((1, N)) / N
    for _t in range(itnum):
        xt = xt @ P
        x += xt
    x = np.array(x).reshape(-1)
    x = x / np.sum(x)
    return x


#
# Constructing a line graph
#
def construct_line_net_adj(A, p=1, q=1, add_source_node=True):
    """Construct the supra-adjacent matrix for the weighted network.

    The random walk process in the Node2Vec is the second order Markov process,
    where a random walker at node i remembers the previously visited node j
    and determines the next node k based on i and j.

    We transform the 2nd order Markov process to the 1st order Markov
    process on supra-nodes. Each supra-node represents a pair of
    nodes (j,i) in the original network, a pair of the previouly
    visited node and current node. We place an edge
    from a supra-node (j,i) to another supra-node (i,k) if the random walker
    can transit from node j, i and to k. The weight of edge is given by the
    unnormalized transition probability for the node2vec random walks.
    """

    if A.getformat() != "csr":
        raise TypeError("A should be in the scipy.sparse.csc_matrix")

    # Preprocessing
    # We efficiently simulate the random walk process using the scipy.csr_matrix
    # data structure. The followings are the preprocess for the simulation.
    A.sort_indices()  # the indices have to be sorted
    A_indptr_csr = A.indptr.astype(np.int32)
    A_indices_csr = A.indices.astype(np.int32)
    A_data_csr = A.data.astype(np.float32)
    A = A.T
    A.sort_indices()  # the indices have to be sorted
    A_indptr_csc = A.indptr.astype(np.int32)
    A_indices_csc = A.indices.astype(np.int32)
    A_data_csc = A.data.astype(np.float32)
    num_nodes = A.shape[0]

    # Make the edge list for the supra-networks
    supra_edge_list, edge_weight = construct_node2vec_supra_net_edge_pairs(
        A_indptr_csr,
        A_indices_csr,
        A_data_csr,
        A_indptr_csc,
        A_indices_csc,
        A_data_csc,
        num_nodes,
        p=p,
        q=q,
        add_source_node=add_source_node,
    )

    # Remove isolated nodes from the supra-network and
    # re-index the ids.
    supra_nodes, supra_edge_list = np.unique(supra_edge_list, return_inverse=True)
    supra_edge_list = supra_edge_list.reshape((edge_weight.size, 2))

    # Each row indicates the node pairs in the original net that constitute the supra node
    src_trg_pairs = (np.vstack(divmod(supra_nodes, num_nodes)).T).astype(int)

    # Construct the supra-adjacency matrix
    supra_node_num = supra_nodes.size
    Aspra = sparse.csr_matrix(
        (edge_weight, (supra_edge_list[:, 0], supra_edge_list[:, 1])),
        shape=(supra_node_num, supra_node_num),
    )
    return Aspra, src_trg_pairs


@numba.jit(nopython=True, cache=True)
def construct_node2vec_supra_net_edge_pairs(
    A_indptr_csr,
    A_indices_csr,
    A_data_csr,
    A_indptr_csc,
    A_indices_csc,
    A_data_csc,
    num_nodes,
    p,
    q,
    add_source_node,
):
    """Construct the weight and edges for the supra adjacency matrix for a
    network, where each supra-node represents a pair of source and target nodes
    in the original network. The supra-edge represents the edge weight for the
    node2vec biased random walk process.

    Parameters
    ----------
    A_indptr_csr : numpy.ndarray
        A_indptr_csr is given by A.indptr, where A is scipy.sparse.csr_matrix
        representing the adjacency matrix for the original network
    A_indices_csr : numpy.ndarray
        A_indices_csr is given by A.indices, where A is scipy.sparse.csr_matrix
        as in A_indptr_csr
    A_data_csr : numpy.ndarray
        A_data_csr is given by A.data, where A is the scipy.sparse.csr_matrix
        as in A_indptr_csr
    A_indptr_csc : numpy.ndarray
        A_indptr_csc is given by A.indptr, where A is scipy.sparse.csc_matrix
        representing the adjacency matrix for the original network
    A_indices_csc : numpy.ndarray
        A_indices_csc is given by A.indices, where A is scipy.sparse.csc_matrix
        as in A_indptr_csc
    A_data_csc : numpy.ndarray
        A_data_csc is given by A.data, where A is the scipy.sparse.csc_matrix
        as in A_indptr_csc
    num_nodes : int
        Number of nodes in the original network
    p : float
        Parameter for the biased random walk. A smaller value encourages
        the random walker to return to the previously visited node.
    q : float
        Parameter for the biased random walk. A smaller value encourages the
        random walker to go away from the previously visited node.

    Return
    ------
    supra_edge_list : np.ndarray
        Edge list for the supra nodes. The id of the node is given by
        source * num_node + target, where source and target is the
        ID of the node in the original network.
    edge_weight_list : np.ndarray
        Edge weight for the edges.
    """

    num_edges = 0
    for i in range(num_nodes):
        outdeg = A_indptr_csr[i + 1] - A_indptr_csr[i]
        indeg = A_indptr_csc[i + 1] - A_indptr_csc[i]
        num_edges += (
            outdeg * indeg
        )  # number of paths of length 2 intermediated by node i
        if add_source_node:
            # edges emanating from the starting supra-node (i,i)
            num_edges += outdeg

    supra_edge_list = -np.ones((num_edges, 2))
    edge_weight_list = -np.zeros(num_edges)
    edge_id = 0
    for i in range(num_nodes):

        # neighbors for the outgoing edges
        # (i)->(next_node)
        for next_nei_id in range(A_indptr_csr[i], A_indptr_csr[i + 1]):
            next_node = A_indices_csr[next_nei_id]
            edge_w = A_data_csr[next_nei_id]

            # neighbors for the incoming edges
            # (prev_node)->(i)
            for prev_node in A_indices_csc[A_indptr_csc[i] : A_indptr_csc[i + 1]]:
                w = edge_w

                # If the next_node and prev_node are the same
                if next_node == prev_node:
                    w = w / p
                else:
                    # Check if next_node is a common neighbor for (prev_node)
                    # and (i) (prev_out_neighbor)<-(prev_node)->(i)
                    # ->(next_node)
                    is_common_neighbor = False
                    for prev_out_neighbor in A_indices_csr[
                        A_indptr_csr[prev_node] : A_indptr_csr[prev_node + 1]
                    ]:
                        # If True, next_node is not the common neighbor
                        # because prev_out_neighbor is increasing.
                        if next_node < prev_out_neighbor:
                            break
                        if prev_out_neighbor == next_node:  # common neighbor
                            is_common_neighbor = True
                            break
                    if is_common_neighbor is False:
                        w = w / q

                # Add an edge between two supra-nodes, (prev_node, i)
                # and (i, next_node)  weighted by w. The ID of the
                # surpra-node composed of (u,v) is given by
                # u * num_nodes + v
                supra_edge_list[edge_id, 0] = prev_node * num_nodes + i
                supra_edge_list[edge_id, 1] = i * num_nodes + next_node
                edge_weight_list[edge_id] = w
                edge_id += 1

            if add_source_node:
                # Add a node
                # supra-node
                # commposed of
                # (i,i) which
                # othe random walker
                # starts from.
                supra_edge_list[edge_id, 0] = i * num_nodes + i
                supra_edge_list[edge_id, 1] = i * num_nodes + next_node
                edge_weight_list[edge_id] = edge_w
                edge_id += 1

    return supra_edge_list, edge_weight_list


def to_trans_mat(mat):
    """Normalize a sparse CSR matrix row-wise (each row sums to 1) If a row is
    all 0's, it remains all 0's.

    Parameters
    ----------
    mat : scipy.sparse.csr matrix
        Matrix in CSR sparse format
    Returns
    -------
    out : scipy.sparse.csr matrix
        Normalized matrix in CSR sparse format
    """
    denom = np.array(mat.sum(axis=1)).reshape(-1).astype(float)
    return sparse.diags(1.0 / np.maximum(denom, 1e-32), format="csr") @ mat


def const_sparse_mat(r, c, v, N, uniqify=True, min_nonzero_value=0):

    if uniqify:
        rc = pairing(r, c)
        rc, rc_id = np.unique(rc, return_inverse=True)
        v = np.bincount(rc_id, weights=v, minlength=rc.shape[0])
        r, c = depairing(rc)

    if min_nonzero_value > 0:
        s = v >= min_nonzero_value
        r, c, v = r[s], c[s], v[s]
    return sparse.csr_matrix((v, (r, c)), shape=(N, N))


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
    t = (w ** 2 + w) * 0.5
    y = np.round(z - t).astype(np.int64)
    x = np.round(w - y).astype(np.int64)
    return x, y


#
# Generate sentences from walk sequence as the input for gensim
#
def walk2gensim_sentence(walks, window_length):
    sentences = []
    for i in range(walks.shape[0]):
        w = walks[i, :]
        w = w[(~np.isnan(w)) * (w >= 0)]
        sentences += [w.astype(str).tolist()]
    return sentences


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


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


def to_nxgraph(net):
    if sparse.issparse(net):
        return nx.from_scipy_sparse_matrix(net)
    elif "networkx" in "%s" % type(net):
        return net
    elif "numpy.ndarray" == type(net):
        return nx.from_numpy_array(net)


#
# Logarithm
#
def safe_log(A, minval=1e-12):
    if sparse.issparse(A):
        A.data = np.log(np.maximum(A.data, minval))
        return A
    else:
        return np.log(np.maximum(A, minval))


def elementwise_log(A, min_value=1e-12):
    if sparse.issparse(A):
        B = A.copy()
        B.data = safe_log(B.data, min_value)
        return B
    else:
        return safe_log(A, min_value)


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
        (np.ones_like(group_ids), (node_ids, group_ids)), shape=shape,
    )
    U.data = U.data * 0 + 1
    return U


def matrix_sum_power(A, T):
    """Take the sum of the powers of a matrix, i.e., sum_{t=1} ^T A^t.

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
