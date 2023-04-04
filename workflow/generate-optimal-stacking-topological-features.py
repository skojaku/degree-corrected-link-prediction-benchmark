# /*
#  * @Author: Rachith Aiyappa
#  * @Date: 2023-04-03 10:11:01
#  * @Last Modified by: Rachith
#  * @Last Modified time: 2023-04-04 12:31:39
#  */

"""
extracts topological features of networks
"""

from scipy import sparse
import numpy as np
import sys
import pickle as pkl
from OptimalStackingFunctions import *

if "snakemake" in sys.modules:
    input_original_edge_table_file = snakemake.input["input_original_edge_table_file"]
    input_heldout_file = snakemake.input["input_heldout_net_file"]
    input_train_file = snakemake.input["input_train_net_file"]
    output_heldout_feature = snakemake.output["output_heldout_feature"]
    output_train_feature = snakemake.output["output_train_feature"]
else:
    input_file = "../data/"
    output_file = "../data/"


edge_table = pd.read_csv(input_original_edge_table_file)
src, trg = edge_table["src"].values, edge_table["trg"].values
n_nodes = int(np.maximum(np.max(src), np.max(trg)) + 1)
A_orig = sparse.csr_matrix((np.ones_like(src), (src, trg)), shape=(n_nodes, n_nodes))
A_orig = A_orig.todense()

A_ho = sparse.load_npz(input_heldout_file)
# To ensure that the network is undirected and unweighted.
A_ho = A_ho + A_ho.T
A_ho.data = A_ho.data * 0.0 + 1.0
A_ho = A_ho.todense()

A_tr = sparse.load_npz(input_train_file)
# To ensure that the network is undirected and unweighted.
A_tr = A_tr + A_tr.T
A_tr.data = A_tr.data * 0.0 + 1.0
A_tr = A_tr.todense()


#### extract features #####

# for hyperparmater search (based on the training network (a subgraph of the holdout network))
A_diff = A_ho - A_tr
e_diff = sparse.find(sparse.triu(A_diff, 1))  # true candidates
A_ho_aux = -1 * A_ho + 1
ne_ho = sparse.find(sparse.triu(A_ho_aux, 1))  # false candidates
Nsamples = 10000  # number of samples
edge_t_tr, edge_f_tr = get_true_and_false_edges(Nsamples, e_diff, ne_ho)
# reshape list of tuples to 2D matrix
edge_t_tr = lot_to_2dmatrix(edge_t_tr)
edge_f_tr = lot_to_2dmatrix(edge_f_tr)
# get features
df_f_tr = gen_topol_feats(A_orig, A_tr, edge_f_tr)
df_t_tr = gen_topol_feats(A_orig, A_tr, edge_t_tr)

# for training model (based on the holdout network (a subgraph of the actual graph))
A_diff = A_orig - A_ho
e_diff = sparse.find(sparse.triu(A_diff, 1))  # true candidates
A_orig_aux = -1 * A_orig + 1
ne_orig = sparse.find(sparse.triu(A_orig_aux, 1))  # false candidates
Nsamples = 10000  # number of samples
edge_t_ho, edge_f_ho = get_true_and_false_edges(Nsamples, e_diff, ne_orig)
# reshape list of tuples to 2D matrix
edge_t_ho = lot_to_2dmatrix(edge_t_ho)
edge_f_ho = lot_to_2dmatrix(edge_f_ho)
# get features
df_f_ho = gen_topol_feats(A_orig, A_ho, edge_f_ho)
df_t_ho = gen_topol_feats(A_orig, A_ho, edge_t_ho)

#### load dataframes for train and holdout sets ####
df_tr = creat_full_set(df_t_tr, df_f_tr)
df_ho = creat_full_set(df_t_ho, df_f_ho)

with open(output_heldout_feature, "wb") as f:
    pkl.dump(df_ho, f)

with open(output_train_feature, "wb") as f:
    pkl.dump(df_tr, f)
