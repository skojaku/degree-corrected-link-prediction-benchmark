# %%
import numpy as np
from scipy import sparse, special
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import sys


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
if "snakemake" in sys.modules:
    output_file = snakemake.output["output_file"]
else:
    output_file = "../figs/deg-pos-neg-edges.pdf"

# %% generate networks
k0 = 10
m = 10
n_nodes = 10000
alpha = 2 + k0 / m
net = price_model(m, k0, n_nodes)
indeg = np.array(net.sum(axis=0)).reshape(-1)

# %% positive and negative edge sampling
import gnn_tools

edge_table = []
for _ in tqdm(range(1)):
    lp = gnn_tools.LinkPredictionDataset(
        testEdgeFraction=0.1, negative_edge_sampler="uniform"
    )
    lp.fit(net)
    _, _edge_table = lp.transform()
    edge_table.append(_edge_table)
edge_table = pd.concat(edge_table)

pos_edges = (
    edge_table[edge_table["isPositiveEdge"] == 1].drop(columns="isPositiveEdge").values
)
neg_edges = (
    edge_table[edge_table["isPositiveEdge"] == 0].drop(columns="isPositiveEdge").values
)

pos_deg = indeg[pos_edges.reshape(-1)]
neg_deg = indeg[neg_edges.reshape(-1)]

# %% Compute the expected degrere distribution, ppos_k, pneg_k
ks = np.arange(k0, int(np.max(indeg)) + 1)
pk = np.bincount(indeg.astype(int) - k0)
pk = pk / np.sum(pk)

ppos_k = np.insert(np.cumsum(ks * pk / np.sum(ks * pk)), 0, 0)[:-1]
pneg_k = np.insert(np.cumsum(pk.copy()), 0, 0)[:-1]

# %% plot
sns.set_style("white")
sns.set(font_scale=1.6)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(5, 5))


ax = sns.ecdfplot(
    neg_deg, complementary=True, log_scale=(True, True), ax=ax, label="Negative edge", lw = 2
)
ax = sns.ecdfplot(
    pos_deg, complementary=True, log_scale=(True, True), ax=ax, label="Positive edge", lw = 2,
)
# ax = sns.ecdfplot(indeg, complementary=True, log_scale=(True, True), ax=ax)
ax = sns.lineplot(
    x=ks,
    y=1 - ppos_k,
    ax=ax,
    label=r"$p_{\rm pos}(k)$",
    color="#2d2d2d",
    linestyle="--",
    lw=1.5,
)
ax = sns.lineplot(
    x=ks,
    y=1 - pneg_k,
    ax=ax,
    label=r"$p_{\rm neg}(k)$",
    color="#4d4d4d",
    linestyle=":",
    lw=1.5,
)
ax.set_ylabel("CCDF")
ax.set_xlabel(r"Degree, $k$")
ax.legend(frameon=False)
sns.despine()

# save figure
fig.savefig(output_file, bbox_inches="tight", dpi=300)

# %%