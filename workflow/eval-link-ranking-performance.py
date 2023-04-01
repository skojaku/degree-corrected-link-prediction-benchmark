# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-03-28 10:34:47
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-04-01 07:20:04
# %%
from scipy import sparse
import numpy as np
import pandas as pd
import sys

if "snakemake" in sys.modules:
    ranking_score_file = snakemake.input["ranking_score_file"]
    edge_table_file = snakemake.input["edge_table_file"]
    data_name = snakemake.params["data_name"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/"
    output_file = "../data/"

# ========================
# Load
# ========================
Ranking = sparse.load_npz(ranking_score_file)
edge_table = pd.read_csv(edge_table_file)

# ========================
# Preprocess
# ========================
s = edge_table["isPositiveEdge"].values > 0
src, trg = (
    edge_table["src"].values[s].astype(int),
    edge_table["trg"].values[s].astype(int),
)
# ========================
# Preprocess
# ========================
Y = sparse.csr_matrix((np.ones_like(src), (src, trg)), shape=Ranking.shape)
assert np.max(Y.data) == 1

n_nodes = Y.shape[0]
results = []
for n_prediction in [3, 5, 10, 50]:

    # Compute the micro F1
    Ypred = []
    prec, reca, f1 = np.zeros(n_nodes), np.zeros(n_nodes), np.zeros(n_nodes)
    for i in range(Y.shape[0]):

        y = Y.indices[Y.indptr[i] : Y.indptr[i + 1]]
        if len(y) == 0:
            continue

        pred = Ranking.indices[Ranking.indptr[i] : Ranking.indptr[i + 1]]
        scores = Ranking.data[Ranking.indptr[i] : Ranking.indptr[i + 1]]

        # Order by the prediction scores
        order = np.argsort(-scores)
        pred, scores = pred[order], scores[order]

        # Remove low-ranked nodes
        if len(pred) > n_prediction:
            pred = pred[:n_prediction]

        # Calc metric
        tp = np.sum(np.isin(pred, y))

        prec[i] = tp / np.maximum(len(pred), 1)
        reca[i] = tp / np.maximum(len(y), 1)
        f1[i] = 2 * prec[i] * reca[i] / np.maximum(prec[i] + reca[i], 1e-32)

        # Save the ranking for computing the macro F1
        Ypred.append([np.ones_like(pred) * i, pred])

    # micro f1
    micro_prec, micro_reca, micro_f1 = np.mean(prec), np.mean(reca), np.mean(f1)

    # macro f1
    src, trg = np.concatenate([d[0] for d in Ypred]), np.concatenate(
        [d[1] for d in Ypred]
    )
    Ypred = sparse.csr_matrix((np.ones_like(src), (src, trg)), shape=Ranking.shape)
    assert np.max(Ypred.data) == 1
    tp = Ypred.multiply(Y).sum()
    macro_prec = tp / Ypred.sum()
    macro_reca = tp / Y.sum()
    macro_f1 = 2 * macro_prec * macro_reca / np.maximum(macro_prec + macro_reca, 1e-32)

    _results = pd.DataFrame(
        {
            "score": [
                micro_prec,
                micro_reca,
                micro_f1,
                macro_prec,
                macro_reca,
                macro_f1,
            ],
            "metric": [
                f"microPrec@{n_prediction}",
                f"microReca@{n_prediction}",
                f"microF1@{n_prediction}",
                f"macroPrec@{n_prediction}",
                f"macroReca@{n_prediction}",
                f"macroF1@{n_prediction}",
            ],
            "data": data_name,
        }
    )
    results.append(_results)

results = pd.concat(results)

# ========================
# Save
# ========================
results.to_csv(output_file, index=False)
