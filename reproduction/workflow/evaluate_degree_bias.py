# %%
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import roc_auc_score
from scipy import sparse

if "snakemake" in sys.modules:
    net_file = snakemake.input["net_file"]
    edge_table_file = snakemake.input["edge_table_file"]
    pred_score_file = snakemake.input["pred_score_file"]
    nGroups = snakemake.params["parameters"]["nGroups"]
    output_file = snakemake.output["output_file"]
else:
    pred_score_file = "../data/derived/link-prediction/ht09-contact-list/score_basedOn~emb_testEdgeFraction~0.5_sampleId~3_negativeEdgeSampler~degreeBiased_model~dcSBM_dim~64.csv"
    edge_table_file = "../data/derived/link-prediction/ht09-contact-list/edge_table_basedOn~emb_testEdgeFraction~0.5_sampleId~3_negativeEdgeSampler~degreeBiased_model~dcSBM_dim~64.csv"
    net_file = "../data/derived/link-prediction/ht09-contact-list/net_basedOn~emb_testEdgeFraction~0.5_sampleId~3_negativeEdgeSampler~degreeBiased_model~dcSBM_dim~64.npz"
    output_file = "../data/"

pred_score_table = pd.read_csv(pred_score_file)
edge_table = pd.read_csv(edge_table_file)
net = sparse.load_npz(net_file)


deg = net.sum(axis=1).A1

# ========================
# Preprocess
# ========================
src, trg = edge_table["src"].values, edge_table["trg"].values
y, ypred = pred_score_table["y"].values, pred_score_table["ypred"].values

ypred[pd.isna(ypred)] = np.min(ypred[~pd.isna(ypred)])
ypred[np.isinf(ypred)] = np.min(ypred[~np.isinf(ypred)])


# %%
# partition the edges based on the degree of the source or target nodes (percentile)
import numpy as np


def get_deg_groups(x, nGroups):
    order = np.argsort(x)

    groups = np.zeros(len(x))
    groups[order] = np.digitize(
        np.arange(len(order)), np.linspace(0, len(order), nGroups + 1), right=False
    )
    return np.array(groups - 1).astype(int)


deg_groups = get_deg_groups(np.concatenate([deg[src], deg[trg]]), nGroups)
src_groups = deg_groups[: len(src)]
trg_groups = deg_groups[len(src) :]


def calc_hitsk(y, ypred, k):
    order = np.argsort(-ypred)
    y, ypred = y[order], ypred[order]
    return np.mean(y[: np.minimum(k, len(y))])


def calc_mrr(y, ypred):
    order = np.argsort(-ypred)
    y, ypred = y[order], ypred[order]
    rank = np.where(np.where(y == 1))[0][0] + 1
    try:
        rank = np.where(y == 1)[0][0] + 1
        score = 1 / rank
    except:
        score = 0
    return score


deg_src = deg[src]
deg_trg = deg[trg]

score_results = []
for i in range(nGroups):
    y_i = np.concatenate([y[src_groups == i], y[trg_groups == i]])
    if np.all(y_i == 0) or np.all(y_i == 1):
        continue
    ypred_i = np.concatenate([ypred[src_groups == i], ypred[trg_groups == i]])

    aucroc_i = roc_auc_score(y_i, ypred_i)
    deg_min = np.min(
        np.concatenate([deg_src[src_groups == i], deg_trg[trg_groups == i]])
    )
    deg_max = np.max(
        np.concatenate([deg_src[src_groups == i], deg_trg[trg_groups == i]])
    )
    quantile = i / nGroups

    score_results.append(
        {
            "metric": "aucroc",
            "score": aucroc_i,
            "group": i,
            "deg_min": deg_min,
            "deg_max": deg_max,
            "quantile": quantile,
        }
    )

# Hits
for i in range(nGroups):
    y_i = np.concatenate([y[src_groups == i], y[trg_groups == i]])
    ypred_i = np.concatenate([ypred[src_groups == i], ypred[trg_groups == i]])
    if np.all(y_i == 0) or np.all(y_i == 1):
        continue
    deg_min = np.min(
        np.concatenate([deg_src[src_groups == i], deg_trg[trg_groups == i]])
    )
    deg_max = np.max(
        np.concatenate([deg_src[src_groups == i], deg_trg[trg_groups == i]])
    )
    quantile = i / nGroups

    for k in [5, 10, 20, 50, 100, 250]:
        hits_i = calc_hitsk(y_i, ypred_i, k)
        score_results.append(
            {
                "metric": f"hits@{k}",
                "score": hits_i,
                "group": i,
                "deg_min": deg_min,
                "deg_max": deg_max,
                "quantile": quantile,
            }
        )

# MRR
for i in range(nGroups):
    y_i = np.concatenate([y[src_groups == i], y[trg_groups == i]])
    ypred_i = np.concatenate([ypred[src_groups == i], ypred[trg_groups == i]])
    if np.all(y_i == 0) or np.all(y_i == 1):
        continue
    deg_min = np.min(
        np.concatenate([deg_src[src_groups == i], deg_trg[trg_groups == i]])
    )
    deg_max = np.max(
        np.concatenate([deg_src[src_groups == i], deg_trg[trg_groups == i]])
    )
    quantile = i / nGroups
    mrr_i = calc_mrr(y_i, ypred_i)
    score_results.append(
        {
            "metric": "mrr",
            "score": mrr_i,
            "group": i,
            "deg_min": deg_min,
            "deg_max": deg_max,
            "quantile": quantile,
        }
    )

# ========================
# Save
# ========================
df = pd.DataFrame(score_results)
df.to_csv(output_file, index=False)
