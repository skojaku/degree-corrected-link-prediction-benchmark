# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-03-28 10:34:47
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-05 15:26:18
# %%
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import roc_auc_score

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    data_name = snakemake.params["data_name"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/derived/link-prediction/ht09-contact-list/score_basedOn~emb_testEdgeFraction~0.25_sampleId~3_negativeEdgeSampler~uniform_model~node2vec_dim~128.csv"
    data_name = "ht09-contact-list"
    output_file = "../data/"

# ========================
# Load
# ========================
data_table = pd.read_csv(input_file)


# ========================
# Preprocess
# ========================

# Assume y is sorted in descending order of ypred
def calc_hitsk(y, k):
    return np.mean(y[:np.minimum(k, len(y))])

def calc_mrr(y):
    rank = np.where(np.where(y == 1))[0][0] + 1
    try:
        rank = np.where(y == 1)[0][0] + 1
        score = 1/rank
    except:
        score = 0
    return score


y, ypred = data_table["y"].values, data_table["ypred"].values
ypred[pd.isna(ypred)] = np.min(ypred[~pd.isna(ypred)])
ypred[np.isinf(ypred)] = np.min(ypred[~np.isinf(ypred)])

# Sort
order = np.argsort(-ypred)
ypred, y = ypred[order], y[order]


results = []
for topk in [5, 10, 20, 50, 100, 250]:
    score = calc_hitsk(y, topk)
    results.append({"metric": f"Hits@{topk}", "score": score, "data": data_name})

mrr = calc_mrr(y)
results.append({"metric": "MRR", "score": mrr, "data":data_name})


# ========================
# Save
# ========================
df = pd.DataFrame(results)
df.to_csv(output_file, index=False)