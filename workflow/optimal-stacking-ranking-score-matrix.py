# /*
#  * @Author: Rachith Aiyappa 
#  * @Date: 2023-04-24 18:08:38 
#  * @Last Modified by:   Rachith 
#  * @Last Modified time: 2023-04-24 18:08:38 
#  */


from scipy import sparse
import numpy as np
import sys
import pickle as pkl
from OptimalStackingFunctions import *

if "snakemake" in sys.modules:
    model_file = snakemake.input["model_file"]
    input_original_edge_table_file = snakemake.input["input_original_edge_table_file"]
    input_heldout_file = snakemake.input["input_heldout_net_file"]
    input_edge_candidates_file = snakemake.input["input_edge_candidates_file"]
    output_file = snakemake.output["output_file"]
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

with open(input_edge_candidates_file,"rb") as f:
    true_false_candidates = pkl.load(f)
    
#### extract features #####

# for testing model (based on the holdout network (a subgraph of the actual graph))
e_diff = true_false_candidates["positives_test"] # true candidates
ne_orig = true_false_candidates["negatives_test"] # false candidates
edge_t_ho = lot_to_2dmatrix(list(zip(e_diff[0],e_diff[1])))
edge_f_ho = lot_to_2dmatrix(list(zip(ne_orig[0],ne_orig[1])))
# get features
df_f_ho = gen_topol_feats(len(edge_f_ho),A_orig, A_ho, edge_f_ho)
df_t_ho = gen_topol_feats(len(edge_t_ho),A_orig, A_ho, edge_t_ho)

#### with ground truths ####
df_ho = creat_full_set(df_t_ho, df_f_ho)

feature_set = [
        "com_ne",
        "ave_deg_net",
        "var_deg_net",
        "ave_clust_net",
        "num_triangles_1",
        "num_triangles_2",
        "page_rank_pers_edges",
        "pag_rank1",
        "pag_rank2",
        "clust_coeff1",
        "clust_coeff2",
        "ave_neigh_deg1",
        "ave_neigh_deg2",
        "eig_cent1",
        "eig_cent2",
        "deg_cent1",
        "deg_cent2",
        "clos_cent1",
        "clos_cent2",
        "betw_cent1",
        "betw_cent2",
        "load_cent1",
        "load_cent2",
        "ktz_cent1",
        "ktz_cent2",
        "pref_attach",
        "LHN",
        "svd_edges",
        "svd_edges_dot",
        "svd_edges_mean",
        "svd_edges_approx",
        "svd_edges_dot_approx",
        "svd_edges_mean_approx",
        "short_path",
        "deg_assort",
        "transit_net",
        "diam_net",
        "jacc_coeff",
        "res_alloc_ind",
        "adam_adar",
        "num_nodes",
        "num_edges",
    ]

edge_set = ['i' , 'j']

unseen = df_ho
y_unseen = unseen.TP
X_unseen = unseen.loc[:, feature_set]
edges = unseen.loc[:, edge_set]
X_unseen.fillna(X_unseen.mean(), inplace=True)

col_mean = np.nanmean(X_unseen, axis=0)
inds = np.where(np.isnan(X_unseen))
if len(inds[0]) > 0:
    X_unseen[inds] = np.take(col_mean, inds[1])

with open(model_file,"rb") as f:
    dtree_model = pkl.load(f)

#  prediction on test set
dtree_predictions = dtree_model.predict(X_unseen)
dtree_proba = dtree_model.predict_proba(X_unseen)

results = list(zip(edges['i'],edges['j'],dtree_proba[:,1]))

# Create an array of zeros with the desired shape
arr = np.empty((n_nodes,n_nodes))
arr[:] = np.nan

# Fill in the array with the values from the list of tuples
for row_idx, col_idx, val in results:
    arr[row_idx, col_idx] = val
    arr[col_idx, row_idx] = val # symmetric


with open(output_file, "wb") as f:
    np.save(f,arr)
