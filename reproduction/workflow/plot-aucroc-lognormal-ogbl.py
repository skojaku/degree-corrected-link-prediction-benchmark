# %%
import numpy as np
from scipy import sparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import erf
import sys

if "snakemake" in sys.modules:
    netstat_table_file = snakemake.input["netstat_table_file"]
    aucroc_table_file = snakemake.input["aucroc_table_file"]
    output_file = snakemake.output["output_file"]
else:
    netstat_table_file = "../data/stats/network-stats.csv"
    aucroc_table_file = "../data/derived/results_aster/result_auc_roc.csv"
    aucroc_table_file_ext = "../data/derived/results/result_auc_roc.csv"
    output_file = "../figs/auc-roc-log-normal-ogbl.pdf"

data_table = pd.read_csv(netstat_table_file)
auc_table = pd.read_csv(aucroc_table_file)
auc_table_ext = pd.read_csv(aucroc_table_file_ext)
#auc_table = pd.concat([auc_table, auc_table_ext]).reset_index(drop=True)
auc_table = auc_table_ext
auc_table
#  Get the AUC-ROC for the PA
auc_table = auc_table.query(
    "model == 'preferentialAttachment' and negativeEdgeSampler == 'uniform' and testEdgeFraction == 0.25"
)
auc_table = pd.merge(auc_table, data_table, left_on="data", right_on="network")
# auc_table = auc_table[["model", "network", "score", "lognorm_sigma"]]
# %% Compute the AUC-ROC curve

sigma_list = np.linspace(0.7, 3, 100)
aucroc_list = []
xmin, xmax = -100, 100
n_bins = 10000
x = np.linspace(xmin, xmax, n_bins)
dx = np.diff(x, prepend=(xmax - xmin) / n_bins)
for sigma in sigma_list:
    px = np.exp(-(x**2) / 2) / (np.sqrt(2 * np.pi))
    phix = 0.5 * (1 + erf((x - np.sqrt(2) * sigma) / np.sqrt(2)))

    auc = 1 - np.sum(dx * px * phix)
    aucroc_list.append(auc)

expected_auc = pd.DataFrame({"sigma": sigma_list, "aucroc": aucroc_list})


# %% Plot the AUC-ROC curve
sns.set_style("white")
sns.set(font_scale=1.6)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(5.5, 4.5))

sns.lineplot(
    data=expected_auc,
    x=sigma_list,
    y=aucroc_list,
    color="black",
    ls="--",
    ax=ax,
    label="",
)

df = auc_table.query("n_nodes >= 300")
print(df["data"].map(lambda x: x.split("ogbl-")[1]))
dg = df.copy()
dg["data"] = df["data"].map(lambda x: x.split("ogbl-")[1])
sns.scatterplot(
    #data=df,
    data=dg.groupby("data").mean(),
    x="lognorm_sigma",
    y="score",
    style = "data",
    edgecolor="black",
    s=150,
    zorder=10,
    ax=ax,
    label="",
    color=sns.color_palette("bright")[1],
)

ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.95, 0.1), markerscale=1.5)

ax.set_xlabel("Degree heterogeneity ($\sigma$)")
ax.set_ylabel("AUC-ROC of PA")
ax.set_ylim(0.6, 1.05)


sns.despine()

fig.savefig(output_file, bbox_inches="tight", dpi=300)

# %%
