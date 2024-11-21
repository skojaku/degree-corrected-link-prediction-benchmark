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
    splitCriterion = snakemake.params["parameters"]["splitCriterion"]
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


if splitCriterion == "degdeg":
    edge_score = deg[src] * deg[trg]
elif splitCriterion == "degsum":
    edge_score = deg[src] + deg[trg]
elif splitCriterion == "degmax":
    edge_score = np.maximum(deg[src], deg[trg])
elif splitCriterion == "degmin":
    edge_score = np.minimum(deg[src], deg[trg])
else:
    raise ValueError(f"Unknown split criterion: {splitCriterion}")

edge_groups = get_deg_groups(edge_score, nGroups)


# ========================
# Calculate scores
# ========================
score_results = []
for pos_i in range(nGroups):
    for neg_j in range(nGroups):
        edge_score_pos = edge_score[(edge_groups == pos_i) & (y == 1)]
        edge_score_neg = edge_score[(edge_groups == neg_j) & (y == 0)]

        yij = np.concatenate(
            [np.ones(len(edge_score_pos)), np.zeros(len(edge_score_neg))]
        )
        edge_score_ij = np.concatenate([edge_score_pos, edge_score_neg])

        aucroc_ij = roc_auc_score(yij, edge_score_ij)

        score_results.append(
            {
                "metric": "aucroc",
                "score": aucroc_ij,
                "pos_group": pos_i,
                "neg_group": neg_j,
                "n_samples": len(edge_score_pos) + len(edge_score_neg),
                "n_pos_samples": len(edge_score_pos),
                "n_neg_samples": len(edge_score_neg),
            }
        )

    for k in [5, 10, 20, 50, 100, 250]:
        hits_ij = calc_hitsk(yij, edge_score_ij, k)
        score_results.append(
            {
                "metric": f"hits@{k}",
                "score": hits_ij,
                "pos_group": pos_i,
                "neg_group": neg_j,
                "n_samples": len(edge_score_pos) + len(edge_score_neg),
                "n_pos_samples": len(edge_score_pos),
                "n_neg_samples": len(edge_score_neg),
            }
        )

    mrr_ij = calc_mrr(yij, edge_score_ij)
    score_results.append(
        {
            "metric": "mrr",
            "score": mrr_ij,
            "pos_group": pos_i,
            "neg_group": neg_j,
            "n_samples": len(edge_score_pos) + len(edge_score_neg),
            "n_pos_samples": len(edge_score_pos),
            "n_neg_samples": len(edge_score_neg),
        }
    )

# ========================
# Save
# ========================
df = pd.DataFrame(score_results)
df.to_csv(output_file, index=False)
