# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-03-28 10:34:47
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-05 06:20:50
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
    input_file = "../data/"
    output_file = "../data/"

# ========================
# Load
# ========================
data_table = pd.read_csv(input_file)

# ========================
# Preprocess
# ========================

y, ypred = data_table["y"].values, data_table["ypred"].values
ypred[pd.isna(ypred)] = np.min(ypred[~pd.isna(ypred)])
ypred[np.isinf(ypred)] = np.min(ypred[~np.isinf(ypred)])
aucroc = roc_auc_score(y, ypred)

# ========================
# Save
# ========================
pd.DataFrame({"score": [aucroc], "metric": "aucroc", "data": data_name}).to_csv(
    output_file, index=False
)
