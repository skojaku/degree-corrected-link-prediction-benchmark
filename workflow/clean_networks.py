# -*- coding: utf-8 -*-
# @Author: Rachith Aiyappa
# @Date:   2023-03-28 03:24pm
# @Last Modified by:   Rachith Aiyappa
# @Last Modified time: 2023-03-28 05:44pm

"""This script takes as input as bunch of raw networks, 
and outputs the edge list of its gcc as a csv file."""

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix, triu
import os
import sys
import pandas as pd
import sys

if "snakemake" in sys.modules:
    raw_unprocessed_dir = snakemake.input["raw_unprocessed_networks_dir"]
    #raw_processed_networks_dir = snakemake.input["raw_processed_networks_dir"]
    edge_table_file = snakemake.output["edge_table_file"]
    name_of_network = edge_table_file.split("/")[-2]
else:
    print("executing in standalone manner")
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    raw_unprocessed_dir = "/".join(input_file.split("/")[:-1])
    name_of_network = input_file.split("/")[-1]
    edge_table_file = output_file

for f in os.listdir(raw_unprocessed_dir):
    if name_of_network in f:
        # read as a undirected, unweighted graph. No self loops?
        G = nx.read_edgelist(
            os.path.join(raw_unprocessed_dir, f), create_using=nx.Graph, nodetype=int
        )

        A = nx.to_scipy_sparse_array(G)
        A = csr_matrix(A)

        # following code is part of the
        # backend of get_largest_connected_component in
        # https://scikit-network.readthedocs.io/en/latest/_modules/sknetwork/topology/structure.html#get_connected_components

        # gives the number of connected components and component label for each node.
        n_components, labels = connected_components(
            A, return_labels=True, directed=False
        )
        unique_labels, counts = np.unique(labels, return_counts=True)
        largest_component_label = unique_labels[np.argmax(counts)]
        index = np.argwhere(labels == largest_component_label).ravel()
        Gcc_A = A[index, :]
        Gcc_A = (Gcc_A.tocsc()[:, index]).tocsr()

        break

#os.makedirs(raw_processed_networks_dir + "/" + name_of_network, exist_ok=True)

# returns upper triangle of the matrix inclusive of diagonal to avoid duplicate edges in output
Gcc_A_triup = triu(Gcc_A, k=0, format="csc")
# node labels are integers starting from 0
src = list(Gcc_A_triup.nonzero()[0])
trg = list(Gcc_A_triup.nonzero()[1])

pd.DataFrame({"src": src, "trg": trg}).to_csv(edge_table_file, index=False)
