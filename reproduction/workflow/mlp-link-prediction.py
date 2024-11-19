# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-01-17 04:25:23
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-05 15:26:52
# %%
import numpy as np
import pandas as pd
import sys
from scipy import sparse
import torch
from MLP import load_model

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    net_file = snakemake.input["net_file"]
    model_file = snakemake.input["model_file"]
    model_name = snakemake.params["parameters"]["model"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/"
    output_file = "../data/"

# ========================
# Load
# ========================
edge_table = pd.read_csv(input_file)
net = sparse.load_npz(net_file)
model = load_model(model_file, device="cpu")
model.eval()

# ========================
# Preprocess
# ========================

src, trg, y = (
    edge_table["src"].values.astype(int),
    edge_table["trg"].values.astype(int),
    edge_table["isPositiveEdge"].astype(int),
)

ypred = model.forward_edges(net, src, trg)
ypred = ypred.detach().numpy()

# ========================
# Save
# ========================
pd.DataFrame({"y": y, "ypred": ypred}).to_csv(output_file)
