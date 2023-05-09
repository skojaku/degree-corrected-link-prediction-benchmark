# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-05-05 08:44:53
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-08 21:47:31
# %%
import numpy as np
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from tqdm.auto import tqdm
import ptitprince as pt

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    similarityMetric = snakemake.params["similarityMetric"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/derived/results/result_quantile_ranking.csv"
    output_file = f"../figs/ranking-similarity-similarityMetric~RBO.pdf"
    similarityMetric = "RBO"

# ========================
# Load
# ========================
data_table = pd.read_csv(input_file)
data_table["model"].unique()
# %%
# ========================
# Preprocessing
# ========================
from scipy.stats import spearmanr
import rbo


def calc_spearmanr(rankA, rankB):
    _df = pd.merge(
        pd.DataFrame({"entity": rankA, "scoreA": np.arange(len(rankA))}),
        pd.DataFrame({"entity": rankB, "scoreB": np.arange(len(rankB))}),
        on="entity",
    )
    return spearmanr(_df["scoreA"], _df["scoreB"])[0]


def truncated_RBO(S, T, p=0.9):
    n_S, n_T = len(S), len(T)
    D = np.maximum(n_S, n_T)
    ST = set([])
    denom = 0
    if np.isclose(p, 1):
        weight = 1
    else:
        weight = 1 - p
    retval = 0
    for d in range(1, D + 1):
        if d <= n_S:
            ST.update([S[d - 1]])
        if d <= n_T:
            ST.update([T[d - 1]])

        n_ST = len(ST)
        S_cap_T = 2 * d - n_ST
        retval += weight * float(S_cap_T) / float(d)
        denom += weight
        weight *= p
    return retval / denom


df = data_table.pivot(
    index=["data", "model", "sampleId"], columns=["metric"], values="rank"
).reset_index()
df["model"].unique()
# %%

reference_metric = [c for c in df.columns if "AUCROC" in c]
# print(reference_metric )
focal_metric = [c for c in df.columns if c not in reference_metric + ["data", "model"]]

results = []
for (data, _), dg in tqdm(
    df.groupby(["data", "sampleId"]),
    total=df[["data", "sampleId"]].drop_duplicates().shape[0],
):
    for rm in reference_metric:
        reference_ranking = dg.sort_values(by=rm)["model"].values
        for fm in focal_metric:
            focal_ranking = dg.sort_values(by=fm)["model"].values
            results.append(
                {
                    "data": data,
                    "reference_ranking": rm,
                    "ranking": fm,
                    "metric": "Spearmanr.",
                    "similarity": calc_spearmanr(reference_ranking, focal_ranking),
                }
            )
            results.append(
                {
                    "data": data,
                    "reference_ranking": rm,
                    "ranking": fm,
                    "metric": "RBO",
                    # "similarity": rbo.RankingSimilarity(
                    #    reference_ranking, focal_ranking
                    # ).rbo(p=0.9),
                    "similarity": truncated_RBO(
                        reference_ranking, focal_ranking, p=0.9
                    ),
                }
            )
result_table = pd.DataFrame(results)
# %%
sns.set_style("white")
sns.set(font_scale=1.5)
sns.set_style("ticks")
# fig, ax = plt.subplots(figsize=(7,5))
cmap = sns.color_palette().as_hex()
palette = {}
hue_order = ["Uniform", "Bias\naligned"]
palette = {"Uniform": cmap[0], "Bias\naligned": cmap[1]}
col_order = [
    "macroF1@5",
    "macroF1@10",
    "macroF1@50",
    "macroPrec@5",
    "macroPrec@10",
    "macroPrec@50",
    "macroReca@5",
    "macroReca@10",
    "macroReca@50",
    "microF1@5",
    "microF1@10",
    "microF1@50",
    "microPrec@5",
    "microPrec@10",
    "microPrec@50",
    "microReca@5",
    "microReca@10",
    "microReca@50",
]
similarityMetric = "RBO"


col_order_string = ",".join([f"'{c}'" for c in col_order])
plot_data = result_table.query(
    f"ranking in [{col_order_string}] and metric == '{similarityMetric}'"
)
plot_data["reference_ranking"] = plot_data["reference_ranking"].map(
    {"AUCROC+uniform": "Uniform", "AUCROC+degreeBiased": "Bias\naligned"}
)
plot_data["x"] = plot_data["reference_ranking"].apply(
    lambda x: {"Uniform": -0.7, "Bias\naligned": 1.7}[x]
)
jitter = 0.12
plot_data["x"] = (
    plot_data["x"] + jitter - 2 * jitter * np.random.rand(plot_data.shape[0])
)
g = sns.catplot(
    data=plot_data.sort_values(by="reference_ranking", ascending=False),
    kind="violin",
    x="reference_ranking",
    y="similarity",
    col="ranking",
    bw=0.15,
    col_order=col_order,
    inner="box",
    col_wrap=3,
    cmap=palette,
    # palette = {k:v+"aa" for k, v in palette.items()},
    # color="#efefef",
    width=0.8,
    cut=0,
    alpha=0.8,
    sharey=False,
    sharex=False,
    aspect=0.7,
)
for ax in g.axes.flat:
    for _ax in ax.collections[::2]:
        _ax.set_alpha(0.8)

g.map(
    sns.scatterplot,
    "x",
    "similarity",
    "reference_ranking",
    hue_order=hue_order,
    palette=palette,
    edgecolor=None,
    alpha=0.2,
    size=3,
)
g.set_ylabels("Ranking Bias Overlap")
g.set_xlabels("Sampling method")

g.set_titles(template="{col_name}")
g.fig.savefig(output_file, bbox_inches="tight", dpi=300)

# %%
