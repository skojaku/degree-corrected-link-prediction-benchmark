# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-03-28 10:34:47
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-07-27 10:47:23
# %%
from scipy import sparse
import numpy as np
import pandas as pd
import sys

if "snakemake" in sys.modules:
    ranking_score_file = snakemake.input["ranking_score_file"]
    test_edge_table_file = snakemake.input["test_edge_table_file"]
    data_name = snakemake.params["data_name"]
    output_file = snakemake.output["output_file"]
else:
    ranking_score_file = "../mydata/derived/link-prediction/airport-rach/score_ranking_basedOn~emb_testEdgeFraction~0.5_sampleId~2_model~node2vec_dim~64.npz"
    edge_table_file = "../mydata/derived/datasets/airport-rach/testEdgeTable_testEdgeFraction~0.5_sampleId~2.csv"
    data_name = "test"
    input_file = "../data/"
    output_file = "../data/"

# %% Load
data = np.load(ranking_score_file, allow_pickle=True)
src, trg, score = data["src"], data["trg"], data["score"]

pred_edge_table = pd.DataFrame({"src": src, "trg": trg, "score": score})
test_edge_table = pd.read_csv(test_edge_table_file)


# %% Preprocess
n_nodes = pred_edge_table.shape[0]
src, trg, score = tuple(pred_edge_table[["src", "trg", "score"]].values.T)
Ranking = sparse.csr_matrix(
    (
        np.concatenate([score, score]),
        (np.concatenate([src, trg]), np.concatenate([trg, src])),
    ),
    shape=(n_nodes, n_nodes),
)

# %% ========================
# Preprocess
# ===========================
print(test_edge_table)
src, trg, score = tuple(test_edge_table[["src", "trg", "isPositiveEdge"]].values.T)
src, trg = src[score > 0], trg[score > 0]
Y = sparse.csr_matrix((np.ones_like(src), (src, trg)), shape=Ranking.shape)
assert np.max(Y.data) == 1

n_nodes = Y.shape[0]
results = []
# Compute the micro F1
for n_prediction in [3, 5, 10, 50]:
    Ypred = []
    prec, reca, f1 = [], [], []
    for i in range(Y.shape[0]):
        y = Y.indices[Y.indptr[i] : Y.indptr[i + 1]]
        if len(y) == 0:
            continue

        pred = Ranking.indices[Ranking.indptr[i] : Ranking.indptr[i + 1]]
        scores = Ranking.data[Ranking.indptr[i] : Ranking.indptr[i + 1]]
        # Order by the prediction scores
        order = np.argsort(-scores)
        pred, scores = pred[order], scores[order]

        _pred = pred.copy()
        # Remove low-ranked nodes
        if len(pred) > n_prediction:
            _pred = _pred[:n_prediction]

        # Calc metric
        tp = np.sum(np.isin(_pred, y))

        _prec = tp / np.maximum(len(_pred), 1)
        _reca = tp / np.maximum(len(y), 1)
        _f1 = 2 * _prec * _reca / np.maximum(_prec + _reca, 1e-32)
        prec.append(_prec)
        reca.append(_reca)
        f1.append(_f1)

        # Save the ranking for computing the macro F1
        Ypred.append([np.ones_like(_pred) * i, _pred])

    # macro
    macro_prec, macro_reca, macro_f1 = np.mean(prec), np.mean(reca), np.mean(f1)

    # micro f1
    src, trg = np.concatenate([d[0] for d in Ypred]), np.concatenate(
        [d[1] for d in Ypred]
    )
    Ypred = sparse.csr_matrix((np.ones_like(src), (src, trg)), shape=Ranking.shape)
    assert np.max(Ypred.data) == 1
    tp = Ypred.multiply(Y).sum()
    micro_prec = tp / Ypred.sum()
    micro_reca = tp / Y.sum()
    micro_f1 = 2 * micro_prec * micro_reca / np.maximum(micro_prec + micro_reca, 1e-32)

    _results = pd.DataFrame(
        {
            "score": [
                macro_prec,
                macro_reca,
                macro_f1,
                micro_prec,
                micro_reca,
                micro_f1,
            ],
            "metric": [
                f"macroPrec@{n_prediction}",
                f"macroReca@{n_prediction}",
                f"macroF1@{n_prediction}",
                f"microPrec@{n_prediction}",
                f"microReca@{n_prediction}",
                f"microF1@{n_prediction}",
            ],
            "data": data_name,
        }
    )
    results.append(_results)

results = pd.concat(results)
results
# %% ========================
# Save
# ========================
results.to_csv(output_file, index=False)
