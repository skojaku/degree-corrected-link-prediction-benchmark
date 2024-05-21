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
output_file = "aucroc-pa.pdf"

# %% generate networks
import gnn_tools

lp_data = []
k0_list = np.arange(1, 61)
m = 20
n_nodes = 10000
n_samples = 1
for k0 in k0_list:
    for i in range(n_samples):
        alpha = 2 + k0 / m
        net = price_model(m, k0, n_nodes)

        lp = gnn_tools.LinkPredictionDataset(
            testEdgeFraction=0.1, negative_edge_sampler="uniform"
        )
        lp.fit(net)
        train_net, edge_table = lp.transform()
        indeg = np.array(train_net.sum(axis=0)).reshape(-1)

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
# %%
result_table
# %%
import numpy as np
from scipy import sparse, special
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import igraph
from tqdm import tqdm

# result_table.to_csv("aucscore-degree-exponent.csv", index=False)

# %%
result_table = pd.read_csv("aucscore-degree-exponent.csv")

# %% plot
sns.set_style("white")
sns.set(font_scale=1.5)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(5, 5))

ax = sns.lineplot(
    data=result_table, x="alpha", y="score", ax=ax, lw=1.5, label="Simulation"
)
ax = sns.lineplot(
    data=result_table,
    x="alpha",
    y="score_lower",
    ax=ax,
    color="#2d2d2d",
    ls=":",
    lw=2.0,
    label="Lower bound",
)
ax.set_ylabel("AUC-ROC")
ax.set_xlabel(r"Degree exponent")
ax.legend(frameon=False)
sns.despine()

# save figure
output_file = "aucroc-pa.pdf"
fig.savefig(output_file, bbox_inches="tight", dpi=300)

# %%
