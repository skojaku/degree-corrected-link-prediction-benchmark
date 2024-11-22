# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-07-11 22:08:07
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-25 16:11:17
# %%

import sys
import textwrap

import color_palette as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file_performance = snakemake.output["output_file_performance"]
    output_file_aucesim = snakemake.output["output_file_aucesim"]
    params = snakemake.params
    model_names = snakemake.params["model"]
    title = (
        snakemake.params["title"] if "title" in list(snakemake.params.keys()) else None
    )
    with_legend = (
        str(snakemake.params["with_legend"]) == "True"
        if "with_legend" in list(snakemake.params.keys())
        else "True"
    )
else:
    input_file = (
        "../data/derived/community-detection-datasets/lfr/evaluations/all_scores.csv"
    )
    with_legend = True
    params = {
        "dim": 128,
        "n": 3000,
        "metric": "cosine",
        "k": 25,
        "clustering": "kmeans",
        "score_type": "esim",
        "tau": 2.5,
        "kmax":1000,
        "cmax":1000,
        "model": [
            "dcSBN",
            "fineTunedGIN",
            "dcFineTunedGIN",
            "fineTunedGCN",
            "dcFineTunedGCN",
            "fineTunedGraphSAGE",
            "dcFineTunedGraphSAGE",
            "fineTunedGAT",
            "dcFineTunedGAT",
        ],
    }
    tau = params["tau"]
    "../figs/lfr_perf_curve_n~3000_k~25_tau~3_dim~128_minc~100_maxk~1000_maxc~1000.pdf"
    output_file_performance = f"../figs/lfr_scores_performance_curve_tau~{tau}.pdf"
    output_file_aucesim = f"../figs/lfr_scores_aucroc_tau~{tau}.pdf"

#
# Load
#
data_table = pd.read_csv(input_file)
plot_data = data_table.copy()
for k, v in params.items():
    if k not in plot_data.columns:
        continue
    if not isinstance(v, list):
        v = [v]
    plot_data = plot_data[(plot_data[k].isin(v)) | pd.isna(plot_data[k])]

plot_data["model_type"] = plot_data["model"].apply(lambda x: x.replace("dc", ""))
plot_data["Biased"] = plot_data["model"].apply(
    lambda x: "Degree corrected" if "dc" in x else "Original"
)
plot_data
# %%
#
# Plot
#
model_list = plot_data["model"].unique().tolist()

color_palette = sns.color_palette().as_hex()
baseline_color = sns.desaturate(color_palette[0], 0.3)
model_colors = {
    "fineTunedGAT": baseline_color,
    "fineTunedGCN": baseline_color,
    "fineTunedGraphSAGE": baseline_color,
    "fineTunedGIN": baseline_color,
}
model_markers = {
    "fineTunedGAT": "o",
    "fineTunedGCN": "s",
    "fineTunedGraphSAGE": "v",
    "fineTunedGIN": "d",
}
model_linestyles = {k: (1, 1) for k in model_colors.keys()}
model_marker_size = {k: 10 for k in model_colors.keys()}
for k in list(model_colors.keys()):
    k_capitalized = k[0].upper() + k[1:]
    model_colors["dc" + k_capitalized] = "#%02x%02x%02x" % tuple(
        int(c * 255) for c in sns.desaturate(model_colors[k], 0.5)
    )
    model_colors["dc" + k_capitalized] = color_palette[1]
    model_markers["dc" + k_capitalized] = model_markers[k]
    model_linestyles["dc" + k_capitalized] = None
    model_marker_size["dc" + k_capitalized] = 10

model_edge_color = {
    k: sns.dark_palette(model_colors[k], 3)[0] for k in model_colors.keys()
}  # Using hex code for black for clarity
model_list_order = list(model_colors.keys())

sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

fig, ax = plt.subplots(figsize=(6, 5))

for name in model_list_order:
    color = model_colors[name]
    markeredgecolor = model_edge_color[name]
    df = plot_data[plot_data["model"] == name]
    if color == "white":
        ax = sns.lineplot(
            data=df,
            x="mu",
            y="score",
            dashes=model_linestyles[name],
            hue_order=model_list_order,
            color="black",
            ax=ax,
        )

    ax = sns.lineplot(
        data=df,
        x="mu",
        y="score",
        marker=model_markers[name],
        dashes=model_linestyles[name],
        color=color,
        markeredgecolor=markeredgecolor,
        markersize=model_marker_size[name],
        hue_order=model_list_order,
        label=name,
        ax=ax,
    )
# (dummy,) = ax.plot([0.5], [0.5], marker="None", linestyle="None", label="dummy-tophead")

ax.set_xlabel(r"Mixing rate, $\mu$")

if params["score_type"] == "nmi":
    ax.set_ylabel(r"Normalized Mutual Information")
elif params["score_type"] == "esim":
    ax.set_ylabel(r"Element-centric similarity")

ax.set_ylim(-0.03, 1.05)
ax.set_xlim(0, 1.01)
xtick_loc = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
xtick_labels = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
ax.set_xticks(xtick_loc)
ax.set_xticklabels(xtick_labels)
ax.set_yticks(xtick_loc)
ax.set_yticklabels(xtick_labels)

# current_handles, current_labels = ax.get_legend_handles_labels()
# new_handles = []
# new_labels = []
# for i, l in enumerate(current_labels):
#    new_handles.append(current_handles[i])
#    new_labels.append(model_names[l] if l in model_names else l)

if with_legend:
    lgd = ax.legend(
        # new_handles[::-1],
        # new_labels[::-1],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
    )
else:
    ax.legend().remove()
sns.despine()

title = None
if title is not None:
    ax.set_title(textwrap.fill(title, width=42))

fig.savefig(
    output_file_performance,
    bbox_inches="tight",
    dpi=300,
)

# %%
# for name, df in :

plot_data["model"].unique()
# %%

def get_model_type(model_name):
    s = model_name.replace("FineTuned", "")
    s = s.replace("fineTuned", "")
    s = s.replace("dc", "")
    return s
df = (
    plot_data.sort_values(by=["mu"])
    .groupby(["model", "sample"])
    .apply(lambda x: np.trapz(x["score"], x["mu"]))
    .reset_index()
)
df["model_type"] = df["model"].apply(get_model_type)
df["Biased"] = df["model"].apply(
    lambda x: "Degree-corrected" if "dc" in x else "Original"
)
# %%

sns.set_style("white")
sns.set(font_scale=1.3)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(7, 5))

# Add boxplot first
sns.boxplot(
    data=df,
    x="model_type",
    y=0,
    hue="Biased",
    palette={
        "Degree-corrected": color_palette[1],
        "Original": baseline_color,
    },
    width=0.7,
    ax=ax,
)

# Add swarmplot on top
sns.stripplot(
    data=df,
    x="model_type",
    y=0,
    hue="Biased",
    palette={
        "Degree-corrected": color_palette[1],
        "Original": baseline_color,
    },
    size=6,
    edgecolor="k",
    linewidth=1.0,
    alpha = 0.8,
    dodge=True,
    ax=ax,
    legend=False  # Remove legend for boxplot
)

sns.despine()
ax.set_xlabel("Model")
ax.set_ylabel("AUC of performance curve")
ax.legend(frameon=False)

fig.savefig(output_file_aucesim, bbox_inches="tight", dpi=300)

# %%
