# %%
import pandas as pd
import seaborn as sns
import numpy as np
import sys


if "snakemake" in sys.modules:
    auc_roc_table_file = snakemake.input["auc_roc_table_file"]
    netstat_table_file = snakemake.input["netstat_table_file"]
    output_file = snakemake.output["output_file"]
    output_file_uniform = snakemake.output["output_file_uniform"]
else:
    auc_roc_table_file = "../data/derived/results/result_auc_roc.csv"
    netstat_table_file = "../data/stats/network-stats.csv"
    output_file = "../figs/aucroc.pdf"
    output_file_uniform = "../figs/aucroc_uniform.pdf"

# ========================
# Load
# ========================
data_table = pd.read_csv(
    auc_roc_table_file,
)

netstat_table = pd.read_csv(netstat_table_file)

# ========================
# Preprocessing
# ========================

# %% Compute the relative AUC-ROC score

df = (
    data_table.groupby(["data", "negativeEdgeSampler", "model"])["score"]
    .mean()
    .reset_index()
)
plot_data = df.query("model == 'preferentialAttachment'")[
    ["data", "score", "negativeEdgeSampler"]
].rename(columns={"score": "score_pa"})

plot_data = df.merge(plot_data, on=["data", "negativeEdgeSampler"])
plot_data["relative_score"] = plot_data["score"].values / plot_data["score_pa"].values

# %% Merge with the network stats
plot_data = plot_data.merge(netstat_table, left_on="data", right_on="network")


# Let's focus on the "learning" methods
exclude = [
    "dcGCN",
    "dcGAT",
    "dcGraphSAGE",
    "dcGIN",
]
plot_data = plot_data[~plot_data["model"].isin(exclude)]

# %%
df = netstat_table.sort_values("lognorm_sigma")[["network", "lognorm_sigma"]]
df["data_code"] = np.arange(df.shape[0], dtype="int")

plot_data = plot_data.merge(
    df.drop(columns="lognorm_sigma"), left_on="data", right_on="network"
)

plot_data["negativeEdgeSampler"] = plot_data["negativeEdgeSampler"].map(
    {"uniform": "Uniform", "degreeBiased": "Corrected"}
)
# %%
df = plot_data.query("negativeEdgeSampler == 'Corrected' and model == 'preferentialAttachment'")
df["score"].mean()

# %% ========================
# Plot
# ========================
# Create a scatter plot
import matplotlib.pyplot as plt

sns.set_style("white")
sns.set(font_scale=1.6)
sns.set_style("ticks")

df = plot_data.query("negativeEdgeSampler == 'Corrected'").copy()
df["isPA"] = df["model"] == "preferentialAttachment"
df["data_code"] = 100 * df["data_code"] / df["data_code"].nunique()

fig, ax = plt.subplots(figsize=(5.5, 4.5))

sns.scatterplot(
    data=df.query("isPA == False"),
    x="data_code",
    y="score",
    ax=ax,
    s=40,
    zorder=10,
    color=sns.color_palette(desat=0.5).as_hex()[0] + "44",
    lw=0,
)
sns.scatterplot(
    data=df.query("isPA"),
    x="data_code",
    y="score",
    ax=ax,
    s=80,
    zorder=10,
    color=sns.color_palette("bright").as_hex()[1],
)

ax.set_ylabel("AUC-ROC")
ax.set_xlabel(r"Graphs ")
sns.despine()

ax.set(xlim=(-1, 101), ylim=(0.2, 1.01))
ax.legend().remove()

fig.savefig(output_file, bbox_inches="tight", dpi=300)


# %%
sns.set_style("white")
sns.set(font_scale=1.6)
sns.set_style("ticks")

df = plot_data.query("negativeEdgeSampler == 'Uniform'").copy()
df["isPA"] = df["model"] == "preferentialAttachment"
df["data_code"] = 100 * df["data_code"] / df["data_code"].nunique()


fig, ax = plt.subplots(figsize=(5.5, 4.5))

sns.scatterplot(
    data=df.query("isPA == False"),
    x="data_code",
    y="score",
    ax=ax,
    s=40,
    zorder=10,
    color=sns.color_palette(desat=0.5).as_hex()[0] + "44",
    lw=0,
)
sns.scatterplot(
    data=df.query("isPA"),
    x="data_code",
    y="score",
    ax=ax,
    s=80,
    zorder=10,
    color=sns.color_palette("bright").as_hex()[1],
)

ax.set_ylabel("AUC-ROC")
ax.set_xlabel(r"Graphs")
sns.despine()

ax.set(xlim=(-1, 101), ylim=(0.2, 1.01))
ax.legend().remove()
fig.savefig(output_file_uniform, bbox_inches="tight", dpi=300)


# %%
# df = plot_data.copy()
# df["relative_score"] = np.exp(df["relative_score"].values)
# df = (
#    df.query("negativeEdgeSampler =='Bias aligned' ")[["Data", "relative_score", "model"]]
#    .groupby(["Data", "model"])
#    .mean()
# )
# np.mean(df["relative_score"].values < 0)
#
## %%
# df["relative_score"].min()
#
## %%
#
