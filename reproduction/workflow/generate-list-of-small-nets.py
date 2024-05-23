# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-04-18 11:32:29
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-04-18 11:42:37
# %%
import os
import pandas as pd
import numpy as np
from os.path import join
import networkx as nx
import json

max_n_nodes = 20000

DATA_DIR = "../data"  # set test_data for testing
DERIVED_DIR = join(DATA_DIR, "derived")
NETWORK_DIR = join(DERIVED_DIR, "networks")
RAW_UNPROCESSED_NETWORKS_DIR = join(NETWORK_DIR, "raw")

small_net_list = []
for f in os.listdir(RAW_UNPROCESSED_NETWORKS_DIR):
    filename = join(RAW_UNPROCESSED_NETWORKS_DIR, f)
    G = nx.read_edgelist(filename, create_using=nx.Graph, nodetype=int)
    A = nx.to_scipy_sparse_array(G)
    n_nodes = A.shape[0]
    if n_nodes > max_n_nodes:
        continue

    small_net_list.append(f.split("_")[1].split(".")[0])


with open("small-networks.json", "w") as f:
    json.dump(small_net_list, f, indent=2)

# %%
