# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 15:08:01
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-10 08:46:51
# %%
import sys
import numpy as np
from scipy import sparse
from gnn_tools.models import *
import buddy
import GPUtil
from MLP import train_heldout, save_model, load_model

#
# Input
#
if "snakemake" in sys.modules:
    netfile = snakemake.input["net_file"]
    output_file = snakemake.output["output_file"]
    parameters = snakemake.params["parameters"]
    with_degree = parameters["withDegree"]
    negativeEdgeSampler = parameters["trainNegativeEdgeSampler"]
else:
    netfile = "../data/derived/datasets/airport-rach/train-net_testEdgeFraction~0.25_sampleId~0.npz"
    with_degree = False
    negativeEdgeSampler = "uniform"

net = sparse.load_npz(netfile)
net = net + net.T
net.data = net.data * 0.0 + 1.0


def get_gpu_id(excludeID=[]):
    device = GPUtil.getFirstAvailable(
        order="random",
        maxLoad=0.9,
        maxMemory=0.6,
        attempts=99999,
        interval=60 * 1,
        verbose=False,
        # excludeID=excludeID,
        # excludeID=[6, 7],
    )[0]
    device = f"cuda:{device}"
    return device


device = get_gpu_id()

model, val_score = train_heldout(
    network=net,
    with_degree=with_degree,
    device=device,
    negative_edge_sampler=negativeEdgeSampler,
)
model.eval()

save_model(model, output_file)

# %%
model
