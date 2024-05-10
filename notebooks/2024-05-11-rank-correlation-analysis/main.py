# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
import rbo

retrieval_result_file = "../../data/derived/results/result_retrieval.csv"
retrieval_task_data_table = pd.read_csv(retrieval_result_file)

classification_result_file = "../../data/derived/results/result_auc_roc.csv"
classification_task_data_table = pd.read_csv(classification_result_file)


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


exclude = ["dcGIN", "dcGAT", "dcGraphSAGE", "dcGCN"]

classification_task_data_table = classification_task_data_table[
    ~classification_task_data_table["model"].isin(exclude)
]
retrieval_task_data_table = retrieval_task_data_table[
    ~retrieval_task_data_table["model"].isin(exclude)
]

rank_class = rank_classification_task(classification_task_data_table)
rank_retrieval = rank_retrieval_task(retrieval_task_data_table, "vp", 50)
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
        score = rbo.RankingSimilarity(S, T).rbo_ext(p=0.5)
        results.append({"rbo_score": score, "data": data, "sampling": sampling})

result_table = pd.DataFrame(results)
result_table["sampling"] = result_table["sampling"].map(
    {"uniform": "Uniform", "degreeBiased": "Degree-corrected"}
)
result_table
# %%

sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(4, 5))

color_palette = sns.color_palette("bright")
baseline_color = sns.desaturate(color_palette[0], 0.3)

import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.stripplot(
    data=result_table,
    y="rbo_score",
    x="sampling",
    color="#ffffff",
    edgecolor="black",
    linewidth=0.5,
    ax=ax,
)
ax = sns.violinplot(
    data=result_table,
    y="rbo_score",
    x="sampling",
    hue="sampling",
    cut=0.01,
    palette={"Uniform": baseline_color, "Degree-corrected": color_palette[1]},
)
ax.set_ylabel("Rank correlation (Rank-biased Overlap)")
ax.set_xlabel("Sampling type")
sns.despine()

fig.savefig(output_file, bbox_extra_artists=(lgd,), bbox_inches="tight", dpi=300)

# %%
