# /*
#  * @Author: Rachith Aiyappa
#  * @Date: 2023-04-03 09:43:13
#  * @Last Modified by: Rachith
#  * @Last Modified time: 2023-04-04 12:34:26
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

if "snakemake" in sys.modules:
    edge_table_file = snakemake.input["edge_table_file"]
    parameters = snakemake.params["parameters"]
    output_train_net_file = snakemake.output["output_train_net_file"]
    output_heldout_net_file = snakemake.output["output_heldout_net_file"]
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
    testEdgeFraction=parameters["testEdgeFraction"],
    negative_edge_sampler="uniform",
)

model_ho.fit(A_orig)
A_ho, _ = model_ho.transform()

# =====================================================================
# Construct the training network (A_tr) from the heldout network (A_ho)
# This training network is used to choose the
# hyperparmeters of the random forest classifier

# These hyperparameters are then used to train on A_ho.
# =====================================================================

model_tr = LinkPredictionDataset(
    testEdgeFraction=parameters["testEdgeFraction"],
    negative_edge_sampler=parameters["negativeEdgeSampler"],
)
model_tr.fit(A_ho)
A_tr, _ = model_tr.transform()


# ===============
# Save
# ===============
sparse.save_npz(output_heldout_net_file, A_ho)
sparse.save_npz(output_train_net_file, A_tr)
