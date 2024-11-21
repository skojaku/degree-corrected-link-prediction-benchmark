# @Author: Rachith Aiyappa
# @Date: 2024-05-10 12:10:12
# %%
import numpy as np
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import os
import color_palette

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/derived/results/result_auc_roc.csv"
    output_file = "../figs/rank-change.pdf"


def rankchangechart(
    df,
    show_rank_axis=False,
    rank_axis_distance=1.1,
    ax=None,
    scatter=False,
    holes=False,
    line_args={},
    scatter_args={},
    hole_args={},
):
    """
    Inspired from https://github.com/kartikay-bagla/bump-plot-python
    - df: DataFrame with columns as methods (y labels) and two rows indicating their rank in uniform sampling and in degree biased sampling
    - show_rank_axis: Shows the ranks on the far right of the plot
    - ax: Matplotlib axis object
    - scatter: Whether to show scatter points
    - holes: Whether to show see-through holes
    - line_args: Arguments for the line plot
    - scatter_args: Arguments for the scatter plot
    - hole_args: Arguments for the see-through holes
    """

    # figure size

    if ax is None:
        plt.Figure(figsize=(10, 5))
        left_yaxis = plt.gca()
    else:
        left_yaxis = ax

    # label of left y axis
    left_yaxis.set_ylabel("HeaRT", rotation=0, fontsize=15)
    left_yaxis.yaxis.set_label_coords(-0.2, 1.05)

    # Creating and labelling the right axis.
    right_yaxis = left_yaxis.twinx()
    right_yaxis.set_ylabel("biased sampling", rotation=0, fontsize=15)
    right_yaxis.yaxis.set_label_coords(1.18, 1.1)

    axes = [left_yaxis, right_yaxis]

    # Creating the far right axis if show_rank_axis is True
    if show_rank_axis:
        far_right_yaxis = left_yaxis.twinx()
        axes.append(far_right_yaxis)

    #  Compute the rank difference
    x = df_rank.T.values
    dr = x[:, 1] - x[:, 0]
    models = df.columns
    model_rank_down = models[dr <= -8]
    model_rank_up = models[dr >= 8]

    for col in df.columns:
        y = df[col]
        x = df.index.values
        # Plotting blank points on the right axis/axes
        # so that they line up with the left axis.
        for axis in axes[1:]:
            axis.plot(x, y, alpha=0)

        # Drop in rank greater than 8 places
        if col in model_rank_down:
            left_yaxis.plot(
                x,
                y,
                color=sns.color_palette().as_hex()[0],
                **line_args,
                solid_capstyle="round",
            )

        # Increase in rank greater than 8 places
        elif col in model_rank_up:
            left_yaxis.plot(
                x,
                y,
                color=sns.color_palette().as_hex()[3],
                **line_args,
                solid_capstyle="round",
            )

        # all other methods
        else:
            left_yaxis.plot(x, y, color="grey", **line_args, solid_capstyle="round")

        # Adding scatter plots
        if scatter:
            left_yaxis.scatter(x, y, **scatter_args)

            # Adding see-through holes
            if holes:
                bg_color = left_yaxis.get_facecolor()
                left_yaxis.scatter(x, y, color=bg_color, **hole_args)

    # Number of lines
    lines = len(df.columns)

    y_ticks = [*range(1, lines + 1)]

    # Configuring the axes so that they line up well.
    for axis in axes:
        axis.invert_yaxis()
        axis.set_yticks(y_ticks)
        axis.set_ylim((lines + 0.5, 0.5))
        axis.set_xticks([])

    # Sorting the labels to match the ranks.
    left_labels = df.iloc[0].sort_values().index
    right_labels = df.iloc[-1].sort_values().index

    left_labels = [label + " %2d" % (i + 1) for i, label in enumerate(left_labels)]
    right_labels = ["%2d " % (i + 1) + label for i, label in enumerate(right_labels)]

    left_yaxis.set_yticklabels(left_labels)
    right_yaxis.set_yticklabels(right_labels)

    # Setting the position of the far right axis so that it doesn't overlap with the right axis
    if show_rank_axis:
        far_right_yaxis.spines["right"].set_position(("axes", rank_axis_distance))
    # far_right_yaxis.set_ylabel("rank", rotation=0, fontsize=15)
    # far_right_yaxis.yaxis.set_label_coords(2.3, 1.1)

    for pos in ["right", "top", "bottom", "left"]:
        left_yaxis.spines[pos].set_visible(False)
        right_yaxis.spines[pos].set_visible(False)

    left_yaxis.tick_params(axis="both", which="both", length=0)
    right_yaxis.tick_params(axis="both", which="both", length=0)
    return axes


# load results file
results = pd.read_csv(input_file).drop(columns=["Unnamed: 0"])

exclude = ["dcGAT", "dcGCN", "dcGraphSAGE", "dcGIN"]
results = results[~results["model"].isin(exclude)]

model_names = color_palette.get_model_names()
results["model"] = results["model"].map(
    lambda x: model_names[x] if x in model_names else x
)

# sort performances based on average auc score for uniform and biased sampling
sorted_performances_uniform = sorted(
    [
        (k, v)
        for k, v in dict(
            results[results["negativeEdgeSampler"] == "heart"]
            .groupby("model")["score"]
            .mean()
        ).items()
    ],
    key=lambda x: x[1],
)
sorted_performances_biased = sorted(
    [
        (k, v)
        for k, v in dict(
            results[results["negativeEdgeSampler"] == "degreeBiased"]
            .groupby("model")["score"]
            .mean()
        ).items()
    ],
    key=lambda x: x[1],
)

# get ranks of the methods based on average auc score across all networks
uniform_ranks = {}
for i, (model, _) in enumerate(sorted_performances_uniform):
    uniform_ranks[model] = len(sorted_performances_uniform) - i

biased_ranks = {}
for i, (model, _) in enumerate(sorted_performances_biased):
    biased_ranks[model] = len(sorted_performances_biased) - i


#
# create dataframe which is the input to the rank difference chart
# columns are methods, rows are rank for each method based on uniform sampling and biased sampling
df_rank = pd.DataFrame([uniform_ranks, biased_ranks])

# get chart

#
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(3, 6))

rankchangechart(
    df_rank,
    rank_axis_distance=1.8,
    scatter=True,
    holes=False,
    line_args={"linewidth": 4, "alpha": 0.8},
    scatter_args={"color": "#2d2d2d", "zorder": 100},
    hole_args={"s": 7, "alpha": 1},
    ax=ax,
)

# save chart
plt.savefig(output_file, bbox_inches="tight")

# %%
