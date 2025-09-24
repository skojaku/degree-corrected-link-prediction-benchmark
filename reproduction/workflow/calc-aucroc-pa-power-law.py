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
# %%
import networkx as nx
import numpy as np

G = nx.karate_club_graph()
A = nx.adjacency_matrix(G)
labels = np.unique([d[1]["club"] for d in G.nodes(data=True)], return_inverse=True)[1]
deg = A.sum(axis=1)
deg
# %%

a = np.arange(30)
a
# %%
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.sparse as sparse


def preferential_attachment_model_empirical(
    t0, nrefs, net_train, t_start, mu=None, sig=None, c0=20, n0=0
):
    """
    Simulates a preferential attachment model using empirical data.

    Parameters:
    t0 (array-like): Timestamps indicating when each paper was published.
    nrefs (array-like): Number of references each paper contains.
    net_train (scipy.sparse.csr_matrix): Pre-existing citation network used for training.
    t_start (int): The simulation start time.
    mu (float, optional): Mean of the aging function, if applicable. Default is None.
    sig (float, optional): Standard deviation of the aging function, if applicable. Default is None.
    c0 (float, optional): Initial citation count assigned to each paper. Default is 20.
    n0 (int, optional): Minimum number of papers required before starting the simulation. Default is 0.

    Returns:
    scipy.sparse.csr_matrix: A sparse matrix representing the generated citation network.
    """

    n_nodes = len(t0)
    if net_train is not None:
        ct = np.array(net_train.sum(axis=0)).reshape(-1)
        _, trg, _ = sparse.find(net_train)
        ct = np.bincount(trg, minlength=n_nodes)
    else:
        ct = np.zeros(n_nodes, dtype=float)
    ct = ct + np.ones(n_nodes, dtype=float) * c0

    citing_list, cited_list = [], []

    # Aging function and likelihood
    aging_func = lambda t, t_0: np.exp(
        -((np.log(t - t_0) - mu) ** 2) / (2 * sig**2)
    ) / ((t - t_0) * sig * np.sqrt(2 * np.pi))
    with_aging = (mu is not None) and (sig is not None)

    n_appeared = 0
    for t in tqdm(np.sort(np.unique(t0[~pd.isna(t0)]))):
        if t < t_start:
            continue

        # citable papers
        citable = np.where(t0 < t)[0]
        if len(citable) == 0:
            continue

        new_paper_ids = np.where(t0 == t)[0]
        n_appeared += len(new_paper_ids)

        if n_appeared < n0:
            continue

        citing_papers = new_paper_ids[nrefs[new_paper_ids] > 0]
        if len(citing_papers) == 0:
            continue

        pcited = ct[citable].copy()
        if with_aging:
            pcited *= aging_func(t, t0[citable])
        pcited = np.maximum(pcited, 1e-32)
        pcited /= np.sum(pcited)

        nrefs_citing_papers = nrefs[citing_papers]
        cited = np.random.choice(
            citable, p=pcited, size=int(np.sum(nrefs_citing_papers))
        ).astype(int)
        citing = np.concatenate(
            [
                citing_papers[j] * np.ones(int(nrefs_citing_papers[j]))
                for j in range(len(citing_papers))
            ]
        ).astype(int)
        citing_list += citing.tolist()
        cited_list += cited.tolist()
        ct += np.bincount(cited, minlength=len(ct))
        ct[citing_papers] += nrefs[citing_papers]

    if train_net is not None:
        src, trg, _ = sparse.find(train_net)
        citing_list += src.tolist()
        cited_list += trg.tolist()

    citing = np.array(citing_list)
    cited = np.array(cited_list)

    net = sparse.csr_matrix(
        (np.ones_like(citing), (citing, cited)), shape=(n_nodes, n_nodes)
    )

    return net


##
## Exampe code:
##
##
#import networkx as nx  #
#
#train_net = nx.adjacency_matrix(nx.karate_club_graph())
#train_net = sparse.triu(train_net)  # to make it directional
#
## t0 specifies the timestamp of the publication of the papers
#t0 = list(range(train_net.shape[0]))  # 0, 1, 2, ...
#nrefs = list(
#    np.array(train_net.sum(axis=0)).reshape(-1)
#)  # Number of references a paper has
#t_start = np.max(t0) + 1
#
## Add new papers
#t0 = (
#    t0 + [np.max(t0) + 1] * 5
#)  # 5 papers published after the last paper in the training set
#nrefs = nrefs + [3, 1, 2, 3, 2]  # Number of references for the new papers
#
#t0 = (
#    t0 + [np.max(t0) + 2] * 10
#)  # 10 papers published in the second year from the last paper in the training set
#nrefs = nrefs + [1] * 10  # Number of references for the new papers
#
#t0 = (
#    t0 + [np.max(t0) + 3] * 15
#)  # 15 papers published in the third year from the last paper in the training set
#nrefs = nrefs + [3] * 15  # Number of references for the new papers
#
#t0 = np.array(t0)  # Since my code assumes that t0 is an array
#nrefs = np.array(nrefs)  # Since my code assumes that nrefs is an array
#pred_net = preferential_attachment_model_empirical(
#    t0=t0, nrefs=nrefs, net_train=train_net, t_start=t_start, c0=5
#)
#import seaborn as sns
#
#sns.heatmap(pred_net.toarray())
# %%
