# %%
# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 15:08:01
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-04-19 21:56:17
from scipy import sparse
import numpy as np
from tqdm import tqdm
from numba import jit, prange

topology_models = {}
topology_model = lambda f: topology_models.setdefault(f.__name__, f)


@topology_model
def preferentialAttachment(network, src=None, trg=None, maxk=None):
    deg = np.array(network.sum(axis=1)).reshape(-1)
    n_nodes = len(deg)
    if src is None and trg is None:
        assert maxk is not None, "maxk must be specified"
        scores = np.outer(deg, np.sort(deg)[::-1][:maxk])
        indices = np.outer(np.ones(n_nodes, dtype=int), np.argsort(deg)[::-1][:maxk])
        return scores, indices

    return deg[src] * deg[trg]


@topology_model
def commonNeighbors(network, src=None, trg=None, maxk=None):
    if src is None and trg is None:
        assert maxk is not None, "maxk must be specified"
        S = network @ network
        S = S - S.multiply(network)
        scores, indices = find_k_largest_elements(S, maxk)
        return scores, indices
    return np.array((network[src, :].multiply(network[trg, :])).sum(axis=1)).reshape(-1)


@topology_model
def jaccardIndex(network, src=None, trg=None, maxk=None):
    deg = np.array(network.sum(axis=1)).reshape(-1)

    if src is None and trg is None:
        assert maxk is not None, "maxk must be specified"
        S = network @ network
        S = S - S.multiply(network)
        s, r, v = sparse.find(S)
        v = v / np.maximum(deg[s] + deg[r] - v, 1)
        S = sparse.csr_matrix((v, (s, r)), shape=S.shape)
        S = S - S.multiply(network)

        scores, indices = find_k_largest_elements(S, maxk)
        return scores, indices

    score = np.array((network[src, :].multiply(network[trg, :])).sum(axis=1)).reshape(
        -1
    )
    return score / np.maximum(deg[src] + deg[trg] - score, 1)


@topology_model
def resourceAllocation(network, src=None, trg=None, maxk=None):
    deg = np.array(network.sum(axis=1)).reshape(-1)
    deg_inv = 1 / np.maximum(deg, 1)
    deg_inv[deg == 0] = 0
    if src is None and trg is None:
        assert maxk is not None, "maxk must be specified"
        S = network @ sparse.diags(deg_inv) @ network
        S = S - S.multiply(network)
        scores, indices = find_k_largest_elements(S, maxk)
        return scores, indices
    return np.array(
        ((network[src, :] @ sparse.diags(deg_inv)).multiply(network[trg, :])).sum(
            axis=1
        )
    ).reshape(-1)


@topology_model
def adamicAdar(network, src=None, trg=None, maxk=None):
    deg = np.array(network.sum(axis=1)).reshape(-1)
    log_deg_inv = 1 / np.maximum(np.log(np.maximum(deg, 1)), 1)
    log_deg_inv[deg == 0] = 0
    if src is None and trg is None:
        assert maxk is not None, "maxk must be specified"
        S = network @ sparse.diags(log_deg_inv) @ network
        S = S - S.multiply(network)
        scores, indices = find_k_largest_elements(S, maxk)
        return scores, indices
    return np.array(
        ((network[src, :] @ sparse.diags(log_deg_inv)).multiply(network[trg, :])).sum(
            axis=1
        )
    ).reshape(-1)


@topology_model
def localRandomWalk(network, src=None, trg=None, maxk=None, batch_size=10000):
    deg = np.array(network.sum(axis=1)).reshape(-1)
    deg_inv = 1 / np.maximum(deg, 1)
    P = sparse.diags(deg_inv) @ network
    if src is None and trg is None:
        assert maxk is not None, "maxk must be specified"
        if network.shape[0] > 10000:
            scores, indices = localRandomWalkBatch(network, maxk, batch_size)
            return scores, indices
        PP = P @ P
        PPP = PP @ P
        S = P + PP + PPP
        S = S - S.multiply(network)
        scores, indices = find_k_largest_elements(S, maxk)
        return scores, indices
    else:

        def batch_local_random_walk(src_batch, trg_batch):
            usrc, src_uids = np.unique(src_batch, return_inverse=True)
            PP = P[usrc, :] @ P
            PPP = PP @ P
            S = P[usrc, :] + PP + PPP
            S = sparse.diags(deg[usrc] / np.sum(deg)) @ S
            return np.array(S[(src_uids, trg_batch)]).reshape(-1)

        batch_size = np.minimum(len(src), batch_size)
        results, results_edge_ids = [], []
        usrc = np.unique(src)
        for start in range(0, len(usrc), batch_size):
            end = min(start + batch_size, len(usrc))
            focal_edge_ids = np.where(np.isin(src, usrc[start:end]))[0]
            trg_batch = trg[focal_edge_ids]
            src_batch = src[focal_edge_ids]

            results.append(batch_local_random_walk(src_batch, trg_batch))
            results_edge_ids.append(focal_edge_ids)

        order = np.argsort(np.concatenate(results_edge_ids))
        return np.concatenate(results)[order]


def localRandomWalkBatch(train_net, maxk, batch_size=None):
    n_nodes = train_net.shape[0]
    if batch_size is None:
        batch_size = n_nodes // 100
    n_batches = int(np.ceil(n_nodes / batch_size))

    deg = np.array(train_net.sum(axis=1)).reshape(-1)
    deg_inv = 1 / np.maximum(deg, 1)
    P = sparse.diags(deg_inv) @ train_net
    P_csc = sparse.csc_matrix(P)

    predicted = np.zeros((n_nodes, maxk), dtype=np.int32)
    scores = np.zeros((n_nodes, maxk), dtype=np.float32)
    for i in range(n_batches):
        start = i * batch_size
        end = np.minimum(start + batch_size, n_nodes)
        U = P[start:end, :].toarray()
        P1 = U.copy()
        P2 = P1 @ P_csc  # A @ A
        P3 = P2 @ P_csc  # A @ A + epsilon * A @ A @ A
        batch_net = P1 + P2 + P3
        batch_net = batch_net - U * batch_net
        batch_net[(np.arange(end - start), np.arange(start, end))] = 0
        predicted[start:end] = np.argsort(-batch_net, axis=1)[:, :maxk]
        scores[start:end] = -np.sort(-batch_net, axis=1)[:, :maxk]
    return scores, predicted


def pairing(src, trg):
    return complex(src, trg)


def depairing(pair):
    return pair.real.astype(int), pair.imag.astype(int)


@topology_model
def localPathIndex(
    network, src=None, trg=None, maxk=None, epsilon=1e-3, batch_size=10000
):
    A = network
    if src is None and trg is None:
        assert maxk is not None, "maxk must be specified"
        if network.shape[0] > 10000:
            scores, indices = localPathIndexBatch(network, maxk, epsilon, batch_size)
            return scores, indices
        AA = A @ A
        AAA = AA @ A
        S = AA + epsilon * AAA
        S = S - S.multiply(network)
        S.setdiag(0)
        scores, indices = find_k_largest_elements(S, maxk)
        return scores, indices
    else:

        def batch_local_path_index(src_batch, trg_batch):
            usrc, src_uids = np.unique(src_batch, return_inverse=True)
            AA_usrc = A[usrc, :] @ A
            AAA_usrc = AA_usrc @ A
            S_usrc = AA_usrc + epsilon * AAA_usrc
            return np.array(S_usrc[(src_uids, trg_batch)]).reshape(-1)

        batch_size = np.minimum(len(src), batch_size)
        results, results_edge_ids = [], []
        usrc = np.unique(src)
        for start in range(0, len(usrc), batch_size):
            end = min(start + batch_size, len(usrc))
            focal_edge_ids = np.where(np.isin(src, usrc[start:end]))[0]
            src_batch = src[focal_edge_ids]
            trg_batch = trg[focal_edge_ids]

            results.append(batch_local_path_index(src_batch, trg_batch))
            results_edge_ids.append(focal_edge_ids)

        order = np.argsort(np.concatenate(results_edge_ids))
        return np.concatenate(results)[order]


def localPathIndexBatch(train_net, maxk, epsilon=1e-3, batch_size=None):
    train_net_csc = sparse.csc_matrix(train_net)

    n_nodes = train_net.shape[0]
    if batch_size is None:
        batch_size = n_nodes // 100

    n_batches = int(np.ceil(n_nodes / batch_size))

    predicted = np.zeros((n_nodes, maxk), dtype=np.int32)
    scores = np.zeros((n_nodes, maxk), dtype=np.float32)

    for i in range(n_batches):
        start = i * batch_size
        end = np.minimum(start + batch_size, n_nodes)
        U = train_net[start:end, :].toarray()
        batch_net = U.copy()
        batch_net = batch_net @ train_net_csc  # A @ A
        batch_net = (
            batch_net + epsilon * batch_net @ train_net_csc
        )  # A @ A + epsilon * A @ A @ A
        batch_net = batch_net - U * batch_net
        batch_net[(np.arange(end - start), np.arange(start, end))] = 0
        predicted[start:end] = np.argsort(-batch_net, axis=1)[:, :maxk]
        scores[start:end] = -np.sort(-batch_net, axis=1)[:, :maxk]
    return scores, predicted


def find_k_largest_elements(A, k):
    """A is the scipy csr sparse matrix"""

    scores = np.zeros((A.shape[0], k), dtype=np.float64) * np.nan
    indices = np.zeros((A.shape[0], k), dtype=np.int64) * (-1)
    for i in range(A.shape[0]):
        n_nnz = A.indptr[i + 1] - A.indptr[i]
        ind = np.argsort(-A.data[A.indptr[i] : A.indptr[i + 1]])[: np.minimum(n_nnz, k)]
        indices[i, : len(ind)] = A.indices[A.indptr[i] : A.indptr[i + 1]][ind]
        scores[i, : len(ind)] = A.data[A.indptr[i] : A.indptr[i + 1]][ind]
    return scores, indices

#
# RandomWalk index with forward push
#

@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def _push_local_random_walk(node, residual, walk_scores, neighbors_indptr,
                          neighbors_indices, deg_inv, walk_length):
    """Optimized single node push operation"""
    push_value = residual[node]
    residual[node] = 0
    walk_scores[node] += push_value

    if walk_length > 0:
        start = neighbors_indptr[node]
        end = neighbors_indptr[node + 1]

        if start != end:
            neighbor_update = push_value * deg_inv[node]
            neighbors = neighbors_indices[start:end]
            residual[neighbors] += neighbor_update

    return residual

@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def _forward_push_single_source(source, n, neighbors_indptr, neighbors_indices,
                              deg_inv, walk_scores):
    """Compute walk scores for a single source"""
    for length in range(3):  # Compute 1,2,3 length walks
        residual = np.zeros(n)
        residual[source] = 1.0

        for _ in range(length + 1):
            new_residual = np.zeros(n)
            nonzero_indices = np.where(residual > 0)[0]
            for node in nonzero_indices:
                new_residual = _push_local_random_walk(
                    node, residual, walk_scores[length],
                    neighbors_indptr, neighbors_indices, deg_inv, length
                )
            residual = new_residual

@jit(nopython=True, parallel=True, cache=True, nogil=True, fastmath=True)
def _process_batch(batch_sources, n, neighbors_indptr, neighbors_indices,
                  deg_inv, maxk):
    """Process a batch of source nodes in parallel"""
    batch_size = len(batch_sources)
    batch_scores = np.zeros((batch_size, maxk))
    batch_indices = np.zeros((batch_size, maxk), dtype=np.int32)

    for i in prange(batch_size):
        source = batch_sources[i]
        walk_scores = np.zeros((3, n))

        # Compute walks
        _forward_push_single_source(
            source, n, neighbors_indptr, neighbors_indices,
            deg_inv, walk_scores
        )

        # Combine scores
        combined_scores = walk_scores[0] + walk_scores[1] + walk_scores[2]

        # Remove direct connections and self-loops
        start = neighbors_indptr[source]
        end = neighbors_indptr[source + 1]
        neighbors = neighbors_indices[start:end]
        combined_scores[neighbors] = 0
        combined_scores[source] = 0

        # Get top k efficiently
        top_k_idx = np.argsort(-combined_scores)[:maxk]
        batch_indices[i] = top_k_idx
        batch_scores[i] = combined_scores[top_k_idx]

    return batch_scores, batch_indices

def local_random_walk_forward_push(network, maxk, batch_size=100000):
    """
    Optimized local random walk computation using forward push method.

    Parameters:
    -----------
    network : scipy.sparse.csr_matrix
        Adjacency matrix
    maxk : int
        Number of top scores to return per node
    batch_size : int
        Size of batches for parallel processing

    Returns:
    --------
    scores : ndarray
        Array of shape (n_nodes, maxk) containing top scores
    indices : ndarray
        Array of shape (n_nodes, maxk) containing indices of top scores
    """
    n = network.shape[0]
    maxk = np.minimum(maxk, n - 1)
    network_csr = network.tocsr()

    # Precompute values
    deg = np.array(network_csr.sum(1)).flatten()
    deg_inv = 1.0 / np.maximum(deg, 1e-12)
    neighbors_indptr = network_csr.indptr
    neighbors_indices = network_csr.indices

    # Initialize result arrays
    all_scores = np.zeros((n, maxk))
    all_indices = np.zeros((n, maxk), dtype=np.int32)

    # Process in parallel batches
    n_batches = (n + batch_size - 1) // batch_size
    with tqdm(total=n, desc="Computing local random walks") as pbar:
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n)
            batch_sources = np.arange(start_idx, end_idx)

            # Process batch in parallel
            batch_scores, batch_indices = _process_batch(
                batch_sources, n, neighbors_indptr, neighbors_indices,
                deg_inv, maxk
            )

            # Store results
            all_scores[start_idx:end_idx] = batch_scores
            all_indices[start_idx:end_idx] = batch_indices

            pbar.update(len(batch_sources))

    return all_scores, all_indices