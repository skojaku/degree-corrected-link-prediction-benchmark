# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-07-03 13:21:14
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-07-03 20:41:49
# %%
import itertools
import numpy as np
import pandas as pd
from scipy import sparse
import sys
import GPUtil
from models.LinkPredictionModel import link_prediction_models

if "snakemake" in sys.modules:
    train_net_file = snakemake.input["train_net_file"]
    model_file = snakemake.input["model_file"]
    test_edge_file = snakemake.input["test_edge_file"]
    params = snakemake.params["parameters"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/"
    output_file = "../data/"

# ========================
# Load
# ========================
train_net = sparse.load_npz(train_net_file)
test_edge_table = pd.read_csv(test_edge_file)
model = link_prediction_models[params["modelType"]](**params)
model.load(model_file)

# if "device" not in params:
#    device = GPUtil.getAvailable(
#        order="random",
#        limit=99,
#        maxMemory=0.5,
#        maxLoad=0.5,
#        # excludeID=[0, 5],
#    )[0]
#    device = f"cuda:{device}"
#    params["device"] = device

# ========================
# Preprocess
# ========================
n_nodes = train_net.shape[0]

test_src, test_trg = test_edge_table["src"].values, test_edge_table["trg"].values

# ========================
# Link prediction
# ========================
score = model.predict(train_net, test_src, test_trg)

# ========================
# Save
# ========================
np.savez(output_file, src=test_src, trg=test_trg, score=score)
