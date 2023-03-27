# %%
import embcom
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import sparse
import numpy as np


def load_airport_net():
    # Node attributes
    node_table = pd.read_csv(
        "https://raw.githubusercontent.com/skojaku/core-periphery-detection/master/data/node-table-airport.csv"
    )

    # Edge table
    edge_table = pd.read_csv(
        "https://raw.githubusercontent.com/skojaku/core-periphery-detection/master/data/edge-table-airport.csv"
    )
    # net = nx.adjacency_matrix(nx.from_pandas_edgelist(edge_table))

    net = sparse.csr_matrix(
        (
            edge_table["weight"].values,
            (edge_table["source"].values, edge_table["target"].values),
        ),
        shape=(node_table.shape[0], node_table.shape[0]),
    )

    s = ~pd.isna(node_table["region"])
    node_table = node_table[s]
    labels = node_table["region"].values
    net = net[s, :][:, s]
    return net, labels, node_table

net, labels, node_table = load_airport_net()

# %%
#
# Embedding
#
model = embcom.NonBacktrackingNode2Vec()
model.fit(net)
emb = model.transform(dim = 64)


# %%
# Plot
#
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis(n_components=2)
xy = clf.fit_transform(emb, np.unique(labels, return_inverse=True)[1])


plot_data = pd.DataFrame({"x":xy[:,0], "y":xy[:, 1], "label":labels})

sns.scatterplot(data = plot_data, x = "x", y = "y", hue = "label")


# %%
