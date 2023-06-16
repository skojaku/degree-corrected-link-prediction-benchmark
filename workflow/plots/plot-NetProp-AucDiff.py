# /*
#  * @Author: Rachith Aiyappa
#  * @Date: 2023-03-31 11:14:50
#  * @Last Modified by:   Rachith
#  * @Last Modified time: 2023-03-31 13:12:05
#  */

import os
import sys
import networkx as nx
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import scipy.sparse as sparse
from collections import Counter
from scipy.stats import skew
import seaborn as sns
import matplotlib.pyplot as plt


if "snakemake" in sys.modules:
    results = snakemake.input["auc_results_file"]
    networks = snakemake.input["networks_dir"]
    degskew_outputfile = snakemake.output["degskew_outputfile"]
    nodes_outputfile = snakemake.output["nodes_outputfile"]
    degskew_nodesize_outputfile = snakemake.output["degskew_nodesize_outputfile"]

else:
    print("executing in standalone manner")
    # input_file = sys.argv[1]
    # output_file = sys.argv[2]
    # raw_unprocessed_dir = "/".join(input_file.split("/")[:-1])
    # name_of_network = input_file.split("/")[-1]
    # edge_table_file = output_file

# ==========================================
# Get Network properties of all networks
# (1) skewness of degree distribution skews,
# (2) and number of nodes
# ==========================================
network_names, skews, number_of_nodes = [], [], []
for file in os.listdir(networks):
    edge_table = pd.read_csv(os.path.join(networks, file, "edge_table.csv"))

    # construct network
    src, trg = edge_table["src"].values, edge_table["trg"].values
    n_nodes = int(np.maximum(np.max(src), np.max(trg)) + 1)
    net = sparse.csr_matrix((np.ones_like(src), (src, trg)), shape=(n_nodes, n_nodes))

    # ensure unweighed and undirected
    net = net + net.T
    net.data = net.data * 0.0 + 1.0

    # get degree distribution
    deg_seq = net.sum(axis=1).flatten()
    deg_seq = np.asarray(deg_seq)[0]
    deg_count = dict(Counter(deg_seq))

    # note, for skew calcuclation,
    # it doesn't matter if these are raw counts or probabilities
    # deg_dist = {k:round(v/sum(deg_count.values()),3) for k,v in deg_count.items()}
    # deg_dist = {k:v for k,v in deg_count.items()}

    # get the desired properties
    skews.append(skew(list(deg_count.values())))
    number_of_nodes.append(len(deg_seq))
    network_names.append(file)

network_prop_df = pd.DataFrame(
    {"data": network_names, "degree_skews": skews, "number_of_nodes": number_of_nodes}
)

# ==========================================
# Get the AUC_ROC results
# ==========================================
results_table = pd.read_csv(results)
results_table.rename(columns={"negativeEdgeSampler": "Sampling"}, inplace=True)
results_table["Sampling"] = results_table["Sampling"].map(
    {"uniform": "Uniform", "degreeBiased": "Pref. Attach."}
)

results_table = (
    results_table.groupby(["data", "Sampling", "model"])["score"].mean().reset_index()
)
results_table["score_diff"] = results_table.groupby(["data", "model"])["score"].diff()
results_table = (
    results_table.dropna().reset_index().drop(["Sampling", "score", "index"], axis=1)
)


# ==========================================
# Plots
# ==========================================
plot_data = pd.merge(results_table, network_prop_df, on="data")

sns.set_style("white")
sns.set(font_scale=1)
sns.set_style("ticks")


def annotate(data, **kwargs):
    "function to write out correlation info on plots"
    rp, pp = pearsonr(data[kwargs["x"]], data[kwargs["y"]])
    rs, ps = spearmanr(data[kwargs["x"]], data[kwargs["y"]])
    ax = plt.gca()
    ax.text(
        0.05,
        0.8,
        "Pearson R={:.2f}, p={:.2g} \n\n Spearman R={:.2f}, p={:.2g}".format(
            rp, pp, rs, ps
        ),
        transform=ax.transAxes,
    )


# Degree skew vs auc diff plot
ds_auc = sns.lmplot(
    x="degree_skews", y="score_diff", data=plot_data, col="model", col_wrap=4
)
ds_auc.map_dataframe(annotate, x="degree_skews", y="score_diff")

# Number of nodes vs auc diff plot
ns_auc = sns.lmplot(
    x="number_of_nodes", y="score_diff", data=plot_data, col="model", col_wrap=4
)
ns_auc.map_dataframe(annotate, x="number_of_nodes", y="score_diff")
ns_auc.set(xscale="log")

# Degree skew vs auc diff with size of points indicating number of nodes
ds_auc_nodesize = sns.lmplot(
    x="degree_skews",
    y="score_diff",
    data=plot_data,
    col="model",
    col_wrap=4,
    scatter_kws={"alpha": 0}, #hacky
)
ds_auc_nodesize.map_dataframe(annotate, x="degree_skews", y="score_diff")

ds_auc_nodesize.map(
    sns.scatterplot,
    "degree_skews",
    "score_diff",
    size=plot_data["number_of_nodes"] * 100,
)


# Save
ds_auc.fig.savefig(degskew_outputfile, bbox_inches="tight", dpi=300)
ns_auc.fig.savefig(nodes_outputfile, bbox_inches="tight", dpi=300)
ds_auc.fig.savefig(degskew_nodesize_outputfile, bbox_inches="tight", dpi=300)
