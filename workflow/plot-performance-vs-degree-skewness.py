# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-05-05 11:38:32
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-06 05:50:20
# %%
import numpy as np
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt

if "snakemake" in sys.modules:
    auc_roc_table_file = snakemake.input["auc_roc_table_file"]
    ranking_table_file = snakemake.input["ranking_table_file"]
    net_stat_file = snakemake.input["net_stat_file"]
    output_file = snakemake.output["output_file"]
else:
    auc_roc_table_file = "../data/derived/results/result_auc_roc.csv"
    ranking_table_file = "../data/derived/results/result_ranking.csv"
    net_stat_file = "../data/derived/networks/network-stats.csv"
    output_file = "../figs/performance_vs_degree_kurtosis.pdf"

# ========================
# Load
# ========================
aucroc_table = pd.read_csv(
    auc_roc_table_file,
    usecols=["sampleId", "data", "score", "negativeEdgeSampler", "model"],
).drop_duplicates()
ranking_table = pd.read_csv(
    ranking_table_file, usecols=["sampleId", "data", "score", "metric", "model"]
).drop_duplicates()

net_data_table = pd.read_csv(net_stat_file).rename(columns={"network": "data"})
# %%
# ================================
# Compute the average performance
# ================================
aucroc_table = (
    aucroc_table.groupby(["negativeEdgeSampler", "model", "data"]).mean().reset_index()
)
aucroc_table["metric"] = aucroc_table["negativeEdgeSampler"].apply(
    lambda x: f"AUCROC+{x}"
)
aucroc_table = aucroc_table.drop(columns=["negativeEdgeSampler"])
ranking_table = ranking_table.groupby(["metric", "model", "data"]).mean().reset_index()

data_table = pd.concat([aucroc_table, ranking_table])

data_table = pd.merge(data_table, net_data_table, on="data")
plot_data = data_table.query(
    f"model in ['preferentialAttachment']  and degreeKurtosis>0"
)

# %%
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

import itertools

col_order = ["AUCROC+uniform", "AUCROC+degreeBiased"] + [
    "".join(s)
    for s in itertools.product(
        ["micro", "macro"], ["F1", "Prec", "Reca"], ["@"], ["3", "5", "10", "50"]
    )
]

g = sns.lmplot(
    data=plot_data,
    col="metric",
    x="degreeKurtosis",
    y="score",
    hue="metric",
    col_order=col_order,
    col_wrap=4,
    height=3,
    aspect=1.8,
    palette=[sns.color_palette("Set3").as_hex()[5]],
    order=1,
    logx=True,
    fit_reg=True,
    # lowess=True,
    sharey=False,
    sharex=False,
    scatter_kws={"edgecolor": "#2d2d2d"},
    facet_kws={"gridspec_kws": {"wspace": 0.6}},
)
g.set_titles(template="")
g.set(xscale="log")
g.set(yscale="log")

g.axes[0].set_yscale("linear")
g.axes[1].set_yscale("linear")
g.axes[0].set_ylim(None, 1)
g.axes[1].set_ylim(None, 1)

for i, col in enumerate(col_order):
    g.axes[i].set_ylabel(col)
    g.axes[i].set_xlabel("Degree heterogeneity (kurtosis)")
g.figure.subplots_adjust(wspace=0.4)
g.fig.savefig(output_file, bbox_inches="tight", dpi=300)
