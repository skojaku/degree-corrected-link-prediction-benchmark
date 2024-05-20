# %%
import numpy as np
from scipy import sparse, special
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import igraph
from tqdm import tqdm


# Price model
def price_model(m, k0, n_nodes):

    citing_list, cited_list = [], []
    indeg = np.zeros(n_nodes)

    # Form a star graph
    citing_list += np.arange(1, m).tolist()
    cited_list += np.zeros(m - 1).tolist()
    indeg[0] = m - 1

    for t in tqdm(range(m, n_nodes)):
        prob = indeg[:t] + k0
        prob = prob / np.sum(prob)

        citing = np.ones(m) * t
        cited = np.random.choice(t, p=prob, size=m, replace=False)

        indeg[cited] += 1

        citing_list += citing.tolist()
        cited_list += cited.tolist()

    citing = np.array(citing_list)
    cited = np.array(cited_list)

    _citing = np.concatenate([citing, cited])
    _cited = np.concatenate([cited, citing])

    net = sparse.csr_matrix(
        (np.ones_like(_citing), (_citing, _cited)), shape=(n_nodes, n_nodes)
    )
    return net


# %%
import sys

if "snakemake" in sys.modules:
    output_file = snakemake.output["output_file"]
else:
    output_file = "aucroc-pa.pdf"

# %% generate networks
import gnn_tools

lp_data = []
k0_list = np.arange(1, 61)
m = 20
n_nodes = 10000
n_samples = 10
for k0 in k0_list:
    for i in range(n_samples):
        alpha = 2 + k0 / m
        net = price_model(m, k0, n_nodes)
        indeg = np.array(net.sum(axis=0)).reshape(-1)

        lp = gnn_tools.LinkPredictionDataset(
            testEdgeFraction=0.1, negative_edge_sampler="uniform"
        )
        lp.fit(net)
        _, edge_table = lp.transform()

        alpha = 2 + k0 / m
        lp_data.append(
            {"net": net, "indeg": indeg, "alpha": alpha, "edge_table": edge_table}
        )

# %% positive and negative edge sampling
from sklearn.metrics import roc_auc_score

result_table = []
for dataset in lp_data:
    alpha = dataset["alpha"]
    deg = dataset["indeg"]
    edge_table = dataset["edge_table"]

    score = indeg[edge_table["src"].values] * indeg[edge_table["trg"].values]
    y = edge_table["isPositiveEdge"]

    aucroc = roc_auc_score(y, score)

    score_lower = (alpha - 1) / (2 * alpha - 3)
    score_lower = score_lower**2

    result_table.append({"alpha": alpha, "score": aucroc, "score_lower": score_lower})

result_table = pd.DataFrame(result_table)
result_table.to_csv(output_file, index=False)