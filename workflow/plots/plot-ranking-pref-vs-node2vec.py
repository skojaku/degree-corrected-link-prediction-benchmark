# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-05-05 08:44:53
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-14 04:48:03
# %%
import numpy as np
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    negativeEdgeSampler = snakemake.params["negativeEdgeSampler"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/derived/results/result_quantile_ranking.csv"
    negativeEdgeSampler = "degreeBiased"
    output_file = (
        f"../figs/quantile_ranking_negativeEdgeSampler~{negativeEdgeSampler}.pdf"
    )

# ========================
# Load
# ========================
data_table = pd.read_csv(input_file)
#
# ========================
# Preprocessing
# ========================


# Subsetting and styling
n_models = len(data_table["model"].unique())
plot_data = data_table.copy()
plot_data = (
    plot_data.query("model in ['preferentialAttachment', 'node2vec']")
    .groupby(["data", "model", "metric"])
    .mean()
    .reset_index()
)

# Rename
model2label = {
    "preferentialAttachment": "Pref. Attach.",
    "node2vec": "node2vec",
}


plot_data["model"] = plot_data["model"].map(model2label)

plot_data = plot_data.pivot(
    columns=["metric"], index=["data", "model"], values=["rank"]
)
plot_data.columns = plot_data.columns.get_level_values(1)
plot_data = plot_data.reset_index()
# %%
#
# Plot (unbiased sampling)
#
from matplotlib.patches import Polygon

sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

cmap = sns.color_palette("Set3").as_hex()


def plot(ranking_metric, negativeEdgeSampler, legend, ax):
    margin = n_models * 0.15
    ax.add_patch(
        Polygon(
            [
                [-margin, -margin],
                [n_models + margin, n_models + margin],
                [n_models + margin, -margin],
            ],
            closed=True,
            fill=True,
            facecolor=cmap[3] + "33",
        )
    )
    ax.annotate(
        "Under estimated",
        xy=(1.0, 0.01),
        va="bottom",
        ha="right",
        xycoords="axes fraction",
        fontsize=16,
    )

    ax.add_patch(
        Polygon(
            [
                [-margin, -margin],
                [-margin, n_models + margin],
                [n_models + margin, n_models + margin],
            ],
            closed=True,
            fill=True,
            facecolor=cmap[1] + "33",
        )
    )
    ax.annotate(
        "Over estimated",
        xy=(0.05, 0.98),
        va="top",
        ha="left",
        xycoords="axes fraction",
        fontsize=16,
    )
    ax.plot(
        [-margin, n_models + margin],
        [-margin, n_models + margin],
        ls=":",
        color="#4d4d4d",
    )
    df = (
        plot_data.groupby(
            [f"AUCROC+{negativeEdgeSampler}", f"{ranking_metric}", "model"]
        )
        .size()
        .reset_index(name="sz")
    )
    _cmap = sns.color_palette("dark").as_hex()
    ax = sns.scatterplot(
        data=df,
        # data=plot_data,
        y=f"AUCROC+{negativeEdgeSampler}",
        x=f"{ranking_metric}",
        hue="model",
        style="model",
        style_order=["node2vec", "Pref. Attach."],
        palette={"Pref. Attach.": _cmap[0], "node2vec": _cmap[1]},
        size="sz",
        sizes=(50, 150),
        s=40,
        ax=ax,
    )
    ax.set_xlim(-margin / 3, n_models + margin / 3)
    ax.set_ylim(-margin, n_models + margin)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_ylabel("Rank (AUC-ROC)")
    ax.set_xlabel(f"Rank ({ranking_metric})")
    if legend is False:
        ax.legend().remove()
    # ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncols=2)


sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

ranking_metrics = [
    "".join(s)
    for s in itertools.product(
        ["micro", "macro"], ["F1", "Prec", "Reca"], ["@"], ["3", "5", "10", "50"]
    )
]

g = sns.FacetGrid(
    data=data_table,
    col="metric",
    col_order=ranking_metrics,
    hue_order=["Pref. Attach.", "node2vec"],
    col_wrap=4,
    height=4,
    hue="model",
    sharex=False,
    sharey=False,
)

for i, ax in enumerate(g.axes.flat):
    plot(
        ranking_metrics[i],
        negativeEdgeSampler,
        ax=ax,
        legend=i == len(ranking_metrics) - 3,
    )
g.axes[len(ranking_metrics) - 3].legend(
    loc="upper center",
    bbox_to_anchor=(1.08, -0.15),
    title="",
    ncols=2,
    frameon=False,
    fontsize=17,
)
g.set_titles("")
g.fig.savefig(output_file, bbox_inches="tight", dpi=300)

# %%
