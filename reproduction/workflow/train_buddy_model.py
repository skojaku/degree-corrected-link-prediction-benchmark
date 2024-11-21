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

#
# Input
#
if "snakemake" in sys.modules:
    netfile = snakemake.input["net_file"]
    output_file = snakemake.output["output_file"]
else:
    netfile = "../data/derived/datasets/ogbl-collab/train-net_testEdgeFraction~0.25_sampleId~2.npz"
    output_file = "../data/derived/models/buddy/ogbl-collab/buddy_model~Buddy_testEdgeFraction~0.25_sampleId~2"
    embfile = "tmp.npz"
    params = {"model": "GraphSAGE", "dim": 128}

net = sparse.load_npz(netfile)
net = net + net.T
net.data = net.data * 0.0 + 1.0


def get_gpu_id(excludeID=[]):
    device = GPUtil.getFirstAvailable(
        order="random",
        maxLoad=1.0,
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

print(device)

#
# Embedding
#
# Get the largest connected component
net = sparse.csr_matrix(net)

config = buddy.BuddyConfig()
config.use_feature = False
model, metrics = buddy.train_heldout(
    net, config=config, model_file_path=output_file, device=device
)

# %%
torch.cuda.empty_cache()
# %%
