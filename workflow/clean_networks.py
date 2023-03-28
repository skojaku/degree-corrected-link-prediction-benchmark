# -*- coding: utf-8 -*-
# @Author: Rachith Aiyappa
# @Date:   2023-03-28 03:24pm
# @Last Modified by:   Rachith Aiyappa
# @Last Modified time: 2023-03-28 05:44pm

"""This script takes as input as bunch of raw networks, 
and outputs the edge list of its gcc as a csv file."""

import networkx as nx
import os
import sys
import csv
import pandas as pd

if "snakemake" in sys.modules:
    raw_unprocessed_dir = snakemake.input["raw_unprocessed_networks_dir"]
    raw_processed_networks_dir = snakemake.input["raw_processed_networks_dir"]
    edge_table_file = snakemake.output["edge_table_file"]
else:
    input_file = "../data/"
    output_file = "../data/"

name_of_network = edge_table_file.split("/")[-2]

for f in os.listdir(raw_unprocessed_dir):
    if name_of_network in f:
        # read as a undirected, unweighted graph. No self loops?
        G = nx.read_edgelist(
            os.path.join(raw_unprocessed_dir, f), create_using=nx.Graph, nodetype=int
        )

        gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        Gcc = G.subgraph(gcc[0])

        break

os.makedirs(raw_processed_networks_dir + "/" + name_of_network, exist_ok=True)

# node labels are integers starting from 0
Gcc = nx.convert_node_labels_to_integers(Gcc, first_label=0)
src_trg_list = list(map(list, zip(*Gcc.edges())))
src, trg = src_trg_list[0],src_trg_list[1]

pd.DataFrame({"src":src, "trg":trg}).to_csv(edge_table_file, index=False)