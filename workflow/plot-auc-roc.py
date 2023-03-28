# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-03-28 10:06:41
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-03-28 11:19:06
import numpy as np
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/derived/results/result_auc_roc.csv"
    output_file = "../data/"

# ========================
# Load
# ========================
data_table = pd.read_csv(input_file)

#%% ========================
# Style
# ========================
plot_data = data_table.copy()
plot_data = plot_data.rename(columns={"negativeEdgeSampler": "Sampling"})

plot_data["Sampling"] = plot_data["Sampling"].map(
    {"uniform": "Uniform", "degreeBiased": "Pref. Attach."}
)

palette = {
    "Pref. Attach.": "red",
    "Uniform": "#adadad",
}
# %%
# ========================
# Plot
# ========================
sns.set_style("white")
sns.set(font_scale=1)
sns.set_style("ticks")

g = sns.catplot(
    data=plot_data,
    x="score",
    y="model",
    hue="Sampling",
    col="data",
    row="testEdgeFraction",
    kind="bar",
    palette=palette,
)

g.set_xlabels("AUC-ROC")
g.set_ylabels("Model")

# ========================
# Save
# ========================
g.fig.savefig(output_file, bbox_inches="tight", dpi=300)

# %%
