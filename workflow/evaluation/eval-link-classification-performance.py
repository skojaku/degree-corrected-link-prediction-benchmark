# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-03-28 10:34:47
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-07-28 16:42:25
# %%
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import roc_auc_score

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    data_name = snakemake.params["data_name"]
    test_edge_file = snakemake.input["test_edge_file"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/derived/link-prediction/ht09-contact-list/score_basedOn~emb_testEdgeFraction~0.5_sampleId~3_negativeEdgeSampler~degreeBiased_model~dcSBM_dim~64.csv"
    output_file = "../data/"

# ========================
# Load
# ========================
data = np.load(input_file, allow_pickle=True)
src, trg, ypred = data["src"], data["trg"], data["score"]

test_edge_table = pd.read_csv(test_edge_file)
df = pd.DataFrame({"src": src, "trg": trg, "ypred": ypred})
data_table = pd.merge(test_edge_table, df, on=["src", "trg"], how="left")


# ========================
# Preprocess
# ========================

y, ypred = data_table["isPositiveEdge"].values, data_table["ypred"].values
# %%
ypred[pd.isna(ypred)] = np.min(ypred[~pd.isna(ypred)])
ypred[np.isinf(ypred)] = np.min(ypred[~np.isinf(ypred)])
aucroc = roc_auc_score(y, ypred)

# ========================
# Save
# ========================
pd.DataFrame({"score": [aucroc], "metric": "aucroc", "data": data_name}).to_csv(
    output_file, index=False
)

# %%
