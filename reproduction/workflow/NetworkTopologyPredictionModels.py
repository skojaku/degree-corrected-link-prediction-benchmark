# %%
# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 15:08:01
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-04-19 21:56:17
from scipy import sparse
import numpy as np
from numba import njit

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
def localRandomWalk(network, src=None, trg=None, maxk=None, batch_size=200000):
    deg = np.array(network.sum(axis=1)).reshape(-1)
    deg_inv = 1 / np.maximum(deg, 1)
    P = sparse.diags(deg_inv) @ network
    if src is None and trg is None:
        assert maxk is not None, "maxk must be specified"
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

def pairing(src, trg):
    return complex(src, trg)

def depairing(pair):
    return pair.real.astype(int), pair.imag.astype(int)

@topology_model
def localPathIndex(network, src=None, trg=None, maxk=None, epsilon=1e-3, batch_size=200000):
    A = network
    if src is None and trg is None:
        assert maxk is not None, "maxk must be specified"
        AA = A @ A
        AAA = AA @ A
        S = AA + epsilon * AAA
        S = S - S.multiply(network)
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