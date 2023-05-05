# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-05-05 08:44:53
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-05 11:36:44
import numpy as np
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    negativeEdgeSampler = snakemake.params["parameters"]["negativeEdgeSampler"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "score_quantile_table.csv"
    negativeEdgeSampler = "uniform"

# ========================
# Load
# ========================
data_table = pd.read_csv(input_file)
#
# ========================
# Preprocessing
# ========================


# Subsetting and styling
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
    columns=["metric"], index=["data", "model"], values=["quantile"]
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
    ax.add_patch(
        Polygon(
            [[-0.15, -0.15], [1.15, 1.15], [1.15, -0.15]],
            closed=True,
            fill=True,
            facecolor=cmap[1] + "44",
        )
    )
    ax.annotate(
        "Under evaluation",
        xy=(1.0, 0.01),
        va="bottom",
        ha="right",
        xycoords="axes fraction",
        fontsize=16,
    )

    ax.add_patch(
        Polygon(
            [[-0.15, -0.15], [-0.15, 1.15], [1.15, 1.15]],
            closed=True,
            fill=True,
            facecolor=cmap[3] + "44",
        )
    )
    ax.annotate(
        "Over evaluation",
        xy=(0.05, 0.98),
        va="top",
        ha="left",
        xycoords="axes fraction",
        fontsize=16,
    )
    ax.plot([-0.15, 1.15], [-0.15, 1.15], ls=":", color="#4d4d4d")

    ax = sns.scatterplot(
        data=plot_data,
        y=f"AUCROC+{negativeEdgeSampler}",
        x=f"{ranking_metric}",
        hue="model",
        edgecolor="k",
        s=40,
        alpha=0.8,
        ax=ax,
    )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.15, 1.15)
    ax.set_xlabel(f"Quantile ({ranking_metric})")
    ax.set_ylabel("Quantile (AUC-ROC)")
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
sns.move_legend(
    g.axes[len(ranking_metrics) - 3],
    loc="upper center",
    bbox_to_anchor=(1.08, -0.15),
    title="",
    ncols=2,
    frameon=False,
    fontsize=17,
)
g.set_titles("")
g.fig.savefig(output_file, bbox_inches="tight", dpi=300)
