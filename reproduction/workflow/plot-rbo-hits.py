# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
import rbo


def rank_classification_task(data_table: pd.DataFrame):
    df = data_table.copy()
    df["rank"] = -1
    for (data_name, negativeEdgeSampler), dg in df.groupby(
        ["data", "negativeEdgeSampler"]
    ):
        dg = dg.sort_values(by="score", ascending=False)
        df.loc[dg.index, "rank"] = np.arange(len(dg)) + 1

    df = (
        df[["data", "model", "negativeEdgeSampler", "score", "rank"]]
        .groupby(["data", "negativeEdgeSampler", "model"])
        .mean()
    ).reset_index()
    df["metric"] = "aucroc"
    return df


def rank_retrieval_task(data_table: pd.DataFrame, focal_score: str, topk=50):
    df = data_table.query(f"topk == {topk}")
    df["vp"] = np.maximum(df["prec"], df["rec"])

    df["rank"] = -1
    for data_name, dg in df.groupby("data_name"):
        dg = dg.sort_values(by=focal_score, ascending=False)
        df.loc[dg.index, "rank"] = np.arange(len(dg)) + 1

    df = (
        df[["data_name", "model", focal_score, "rank"]]
        .groupby(["data_name", "model"])
        .mean()
    ).reset_index()
    df["metric"] = focal_score

    return df.rename(columns={"data_name": "data", focal_score: "score"})


import sys

if "snakemake" in sys.modules:
    retrieval_result_file = snakemake.input["retrieval_result"]
    classification_result_file = snakemake.input["classification_result"]
    output_file = snakemake.output["output_file"]
    rbop = float(snakemake.params["rbop"])
    topk = int(snakemake.params["topk"])
    focal_score = snakemake.params["focal_score"]
    metric = snakemake.params["metric"]
else:
    retrieval_result_file = "../data/derived/results/result_retrieval.csv"
    classification_result_file = "../data/derived/results/result_hits-mrr.csv"
    rbop = 0.5
    topk = 10
    focal_score = "vp"
    output_file = "../../figs/rbo.pdf"
    metric = "Hits@50"


retrieval_task_data_table = pd.read_csv(retrieval_result_file)
classification_task_data_table = pd.read_csv(classification_result_file)
classification_task_data_table = classification_task_data_table.query(
    f"metric == '{metric}'"
)

exclude = ["dcGIN", "dcGAT", "dcGraphSAGE", "dcGCN"]

classification_task_data_table = classification_task_data_table[
    ~classification_task_data_table["model"].isin(exclude)
]
retrieval_task_data_table = retrieval_task_data_table[
    ~retrieval_task_data_table["model"].isin(exclude)
]

rank_class = rank_classification_task(classification_task_data_table)
rank_retrieval = rank_retrieval_task(retrieval_task_data_table, focal_score, topk)
# %%

results = []
for sampling, dg in rank_class.groupby("negativeEdgeSampler"):
    dg = dg.rename(columns={"rank": "rank_class", "score": "score_class"})
    df = dg.merge(
        rank_retrieval.rename(
            columns={"rank": "rank_retrieval", "score": "score_retrieval"}
        ),
        on=["data", "model"],
    )
    for data, dh in df.groupby("data"):
        S = dh.sort_values(by="rank_class", ascending=False)["model"].values
        T = dh.sort_values(by="rank_retrieval", ascending=False)["model"].values
        score = rbo.RankingSimilarity(S, T).rbo_ext(p=rbop)
        results.append({"rbo_score": score, "data": data, "sampling": sampling})

result_table = pd.DataFrame(results)
result_table["sampling"] = result_table["sampling"].map(
    {"uniform": "Original", "degreeBiased": "Degree-corrected"}
)

# ===================
# Plot
# ===================
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(4, 5))

color_palette = sns.color_palette().as_hex()
baseline_color = sns.desaturate(color_palette[0], 0.24)
focal_color = sns.color_palette("bright").as_hex()[1]

import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.stripplot(
    data=result_table,
    y="rbo_score",
    x="sampling",
    color="#ffffff",
    edgecolor="black",
    linewidth=0.5,
    hue_order=["Original", "Degree-corrected"],
    order=["Original", "Degree-corrected"],
    ax=ax,
)
ax = sns.violinplot(
    data=result_table,
    y="rbo_score",
    x="sampling",
    hue="sampling",
    bw_adjust=0.5,
    cut=0,
    common_norm=True,
    order=["Original", "Degree-corrected"],
    palette={
        "Original": baseline_color,
        "Degree-corrected": focal_color,
    },
)
ax.set_ylabel("Rank correlation (Rank-biased Overlap)")
ax.set_xlabel("")
sns.despine()

fig.savefig(output_file, bbox_inches="tight", dpi=300)

# %%
