# /*
#  * @Author: Rachith Aiyappa
#  * @Date: 2023-04-03 09:43:13
#  * @Last Modified by: Rachith
#  * @Last Modified time: 2023-04-19 12:44:02 
#  */

"""
creating training and heldout network files
"""

import numpy as np
import pandas as pd
from scipy import sparse
import sys
from tqdm import tqdm
from linkpred.LinkPredictionDataset import LinkPredictionDataset
import pickle as pkl

if "snakemake" in sys.modules:
    edge_table_file = snakemake.input["edge_table_file"]
    edge_fraction_parameters = snakemake.params["edge_fraction_parameters"]
    edge_sampler_parameters = snakemake.params["edge_sampler_parameters"]
    output_train_net_file = snakemake.output["output_train_net_file"]
    output_heldout_net_file = snakemake.output["output_heldout_net_file"]
    output_edge_candidates_file = snakemake.output["output_edge_candidates_file"]
else:
    input_file = "../data/"
    output_file = "../data/"

# =================
# Load
# =================
edge_table = pd.read_csv(edge_table_file)

# ======================================================================
# Construct the heldout network (A_ho) from the original network (A_org)
# ======================================================================
src, trg = edge_table["src"].values, edge_table["trg"].values
n_nodes = int(np.maximum(np.max(src), np.max(trg)) + 1)
A_orig = sparse.csr_matrix((np.ones_like(src), (src, trg)), shape=(n_nodes, n_nodes))

model_ho = LinkPredictionDataset(
    testEdgeFraction=edge_fraction_parameters["testEdgeFraction"],
    negative_edge_sampler=edge_sampler_parameters["negativeEdgeSampler"],
)

model_ho.fit(A_orig)
A_ho, test_edge_table = model_ho.transform()

# =====================================================================
# Construct the training network (A_tr) from the heldout network (A_ho)
# This training network is used to choose the
# hyperparmeters of the random forest classifier
# via 5-fold cross validation on train_edge_table

# These hyperparameters are then used to train a random forest classifier
# on the entire train_edge_table.
# =====================================================================

model_tr = LinkPredictionDataset(
    testEdgeFraction=edge_fraction_parameters["testEdgeFraction"],
    negative_edge_sampler="uniform",
)
model_tr.fit(A_ho)
A_tr, train_edge_table = model_tr.transform()

# ===============
# Save
# ===============
sparse.save_npz(output_heldout_net_file, A_ho)
sparse.save_npz(output_train_net_file, A_tr)

# true and false candidates for training and testing

A_orig = A_orig.todense()
# To ensure that the network is undirected and unweighted.
A_ho = A_ho + A_ho.T
A_ho.data = A_ho.data * 0.0 + 1.0
A_ho = A_ho.todense()

A_tr = A_tr + A_tr.T
A_tr.data = A_tr.data * 0.0 + 1.0
A_tr = A_tr.todense()

true_false_candidates = {}

# for testing model (based on the holdout network (a subgraph of the actual graph))
srcs = test_edge_table[test_edge_table.isPositiveEdge==1].src.values
trgs = test_edge_table[test_edge_table.isPositiveEdge==1].trg.values
A_diff = sparse.csr_matrix((np.ones_like(srcs), (srcs, trgs)), shape=(n_nodes, n_nodes))
positives_test = sparse.find(sparse.triu(A_diff,1)) # true candidates
true_false_candidates['positives_test'] = positives_test

srcs = test_edge_table[test_edge_table.isPositiveEdge==0].src.values
trgs = test_edge_table[test_edge_table.isPositiveEdge==0].trg.values
A_orig_aux = sparse.csr_matrix((np.ones_like(srcs), (srcs, trgs)), shape=(n_nodes, n_nodes))
negatives_test = sparse.find(sparse.triu(A_orig_aux,1)) # false candidates
true_false_candidates['negatives_test'] = negatives_test


# for training model
srcs = train_edge_table[train_edge_table.isPositiveEdge==1].src.values
trgs = train_edge_table[train_edge_table.isPositiveEdge==1].trg.values
A_diff = sparse.csr_matrix((np.ones_like(srcs), (srcs, trgs)), shape=(n_nodes, n_nodes))
positives_train = sparse.find(sparse.triu(A_diff,1)) # true candidates
true_false_candidates['positives_train'] = positives_train

srcs = train_edge_table[train_edge_table.isPositiveEdge==0].src.values
trgs = train_edge_table[train_edge_table.isPositiveEdge==0].trg.values
A_ho_aux = sparse.csr_matrix((np.ones_like(srcs), (srcs, trgs)), shape=(n_nodes, n_nodes))
negatives_train = sparse.find(sparse.triu(A_ho_aux,1)) # false candidates
true_false_candidates['negatives_train'] = negatives_train

with open(output_edge_candidates_file,"wb") as f:
    pkl.dump(true_false_candidates,f)