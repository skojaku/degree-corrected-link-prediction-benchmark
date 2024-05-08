# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-05-03 08:52:00
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-05 15:15:09
# %%
"""
This Python script is a visualization code that reads an edge table
from a CSV file, applies link prediction on it, and generates a scatter
plot with kernel density estimates to visualize the degree distribution
of the edges.
"""
import numpy as np
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse
import gnn_tools

if "snakemake" in sys.modules:
    edge_table_file = snakemake.input["edge_table_file"]
    negativeEdgeSampler = snakemake.params["negativeEdgeSampler"]
    output_file = snakemake.output["output_file"]
else:
    edge_table_file = (
        "../data/derived/networks/preprocessed/airport-rach/edge_table.csv"
    )
    negativeEdgeSampler = "uniform"
    output_file = f"../figs/deg_deg_plot_negativeEdgeSampler~{negativeEdgeSampler}.pdf"

# ========================
# Load
# ========================

edge_table = pd.read_csv(edge_table_file)

src, trg = tuple(edge_table.values.T)

n = int(np.maximum(np.max(src), np.max(trg)) + 1)
net = sparse.csr_matrix((np.ones_like(src), (src, trg)), shape=(n, n))


model = gnn_tools.LinkPredictionDataset(
    testEdgeFraction=0.5, negative_edge_sampler=f"{negativeEdgeSampler}"
)
model.fit(net)
train_net, test_edges = model.transform()

deg = np.array(train_net.sum(axis=0)).reshape(-1)


src, trg, y = tuple(test_edges.values.T)
src, trg = np.concatenate([src, trg]), np.concatenate([trg, src])
y = np.concatenate([y, y])

plot_data = pd.DataFrame(
    {"x": deg[src], "y": deg[trg], "isPositive": np.array(["Negative", "Positive"])[y]}
)
# %%
# Visualization
#
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(5, 5))

ax = sns.kdeplot(
    data=plot_data,
    x="x",
    y="y",
    hue="isPositive",
    hue_order=["Negative", "Positive"],
    palette=sns.color_palette("colorblind")[:2],
    fill=True,
    thresh=2e-1,
    log_scale=(True, True),
    levels=10,
    alpha=0.8,
    edgecolor="k",
    ax=ax,
)
ax = sns.scatterplot(
    data=plot_data.groupby("isPositive").sample(100).sort_values(by="isPositive"),
    x="x",
    y="y",
    hue="isPositive",
    style="isPositive",
    hue_order=["Negative", "Positive"],
    palette=sns.color_palette("colorblind")[:2],
    alpha=1,
    edgecolor="#2d2d2d",
    lw=1,
    ax=ax,
)
ax.set_xlabel("Degree")
ax.set_ylabel("Degree")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncols=2)
sns.despine()

fig.savefig(output_file, bbox_inches="tight", dpi=300)
