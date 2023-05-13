# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 15:08:01
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-05 06:17:31
from scipy import sparse
import numpy as np
import faiss
import pandas as pd
from NetworkTopologyPredictionModels import *
from EmbeddingModels import *


def ranking_by_topology(model_name, network, max_k, batch=1000):
    max_k = np.minimum(network.shape[0] - 1, max_k)

    n = network.shape[1]
    L = int(np.ceil(n / batch))
    score_table = []
    func = topology_models[model_name]
    deg = np.array(network.sum(axis=1)).reshape(-1)
    nodes_sorted_by_degree = np.argsort(-deg)

    src, trg, _ = sparse.find(sparse.triu(network))
    original_src_trg = np.maximum(src, trg) + 1j * np.minimum(src, trg)

    for focal_nodes in np.array_split(np.arange(n), L):
        # Generate the candidates in ranking
        # sort by degree
        if model_name in ["preferentialAttachment"]:
            src = np.kron(focal_nodes, np.ones(max_k)).astype(int)
            trg = np.kron(
                np.ones(len(focal_nodes)),
                nodes_sorted_by_degree[:max_k],
            ).astype(int)
        # check only those with at least common neighbors
        elif model_name in [
            "commonNeighbors",
            "jaccardIndex",
            "resourceAllocation",
            "adamicAdar",
        ]:
            src, trg, _ = sparse.find(network[focal_nodes, :] @ network.T)
            src = focal_nodes[src]

        else:  # Otherwise check all nodes
            n = network.shape[0]
            src = np.kron(focal_nodes, np.ones(n))
            trg = np.kron(np.ones(len(focal_nodes)), np.arange(n))

        _src_trg = np.maximum(src, trg) + 1j * np.minimum(src, trg)
        s = ~np.isin(_src_trg, original_src_trg)
        src, trg = src[s], trg[s]

        _scores = func(network, src, trg)
        df = pd.DataFrame({"query_nodes": src, "value_nodes": trg, "score": _scores})
        df = (
            df.sort_values(by="score", ascending=False)
            .groupby("query_nodes")
            .head(max_k)
        )
        score_table.append(df)
    score_table = pd.concat(score_table)
    return sparse.csr_matrix(
        (
            score_table["score"],
            (score_table["query_nodes"], score_table["value_nodes"]),
        ),
        shape=(n, n),
    )


def ranking_by_embedding(emb, max_k, net, model_name):
    max_k = np.minimum(emb.shape[0] - 1, max_k)
    n = emb.shape[0]
    query_emb = emb.astype("float32").copy()
    key_emb = emb.astype("float32").copy()

    if model_name in ["node2vec", "deepwalk", "graphsage", "line"]:
        query_emb = np.hstack([query_emb, np.ones((n, 1))])
        deg = np.array(net.sum(axis=1)).reshape(-1)
        deg = np.maximum(1, deg)
        deg = deg / np.sum(deg)
        key_emb = np.hstack([key_emb, np.log(deg).reshape((-1, 1))])

    index = make_faiss_index(key_emb, metric="cosine", gpu_id="cpu")
    dist, indices = index.search(query_emb, k=int(max_k))
    scores, value_nodes = np.array(dist).reshape(-1), np.array(indices).reshape(-1)
    query_nodes = np.kron(np.arange(emb.shape[0]), np.ones(max_k)).astype(int)
    s = (query_nodes >= 0) * (value_nodes >= 0)
    scores, query_nodes, value_nodes = scores[s], query_nodes[s], value_nodes[s]
    return sparse.csr_matrix(
        (
            scores,
            (query_nodes, value_nodes),
        ),
        shape=(n, n),
    )


def make_faiss_index(
    X, metric, gpu_id=None, exact=True, nprobe=10, min_cluster_size=10000
):
    """Create an index for the provided data
    :param X: data to index
    :type X: numpy.ndarray
    :raises NotImplementedError: if the metric is not implemented
    :param metric: metric to calculate the similarity. euclidean or cosine.
    :type mertic: string
    :param gpu_id: ID of the gpu, defaults to None (cpu).
    :type gpu_id: string or None
    :param exact: exact = True to find the true nearest neighbors. exact = False to find the almost nearest neighbors.
    :type exact: boolean
    :param nprobe: The number of cells for which search is performed. Relevant only when exact = False. Default to 10.
    :type nprobe: int
    :param min_cluster_size: Minimum cluster size. Only relevant when exact = False.
    :type min_cluster_size: int
    :return: faiss index
    :rtype: faiss.Index
    """
    n_samples, n_features = X.shape[0], X.shape[1]
    X = X.astype("float32")
    if n_samples < 1000:
        exact = True

    index = (
        faiss.IndexFlatL2(n_features)
        if metric == "euclidean"
        else faiss.IndexFlatIP(n_features)
    )

    if not exact:
        nlist = np.maximum(int(n_samples / min_cluster_size), 2)
        faiss_metric = (
            faiss.METRIC_L2 if metric == "euclidean" else faiss.METRIC_INNER_PRODUCT
        )
        index = faiss.IndexIVFFlat(index, n_features, int(nlist), faiss_metric)

    if gpu_id != "cpu":
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, gpu_id, index)
    if not index.is_trained:
        Xtrain = X[
            np.random.choice(
                X.shape[0],
                np.minimum(X.shape[0], min_cluster_size * 5),
                replace=False,
            ),
            :,
        ].copy(order="C")
        index.train(Xtrain)
    index.add(X)
    index.nprobe = nprobe
    return index
