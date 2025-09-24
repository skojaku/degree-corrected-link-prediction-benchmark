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
from LinearClassifier import LinearClassifier

#
# Input
#
if "snakemake" in sys.modules:
    netfile = snakemake.input["net_file"]
    output_file = snakemake.output["output_file"]
    parameters = snakemake.params["parameters"]
    negativeEdgeSampler = parameters["trainNegativeEdgeSampler"]
else:
    netfile = "../data/derived/datasets/airport-rach/train-net_testEdgeFraction~0.25_sampleId~0.npz"
    negativeEdgeSampler = "uniform"

net = sparse.load_npz(netfile)
net = net + net.T
net.data = net.data * 0.0 + 1.0


model = LinearClassifier(negative_edge_sampler=negativeEdgeSampler)
model.train(net)
model.save(output_file)
