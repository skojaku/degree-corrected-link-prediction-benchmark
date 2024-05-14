# %%
import pandas as pd
import seaborn as sns
import numpy as np
import sys


if "snakemake" in sys.modules:
    auc_roc_table_file = snakemake.input["auc_roc_table_file"]
    output_file = snakemake.output["output_file"]
    output_file_uniform = snakemake.output["output_file_uniform"]
else:
    auc_roc_table_file = "../data/derived/results/result_auc_roc.csv"
    output_file = "../figs/log_auc_scatter_plot.pdf"
    output_file_uniform = "../figs/log_auc_scatter_plot_uniform.pdf"


def auc_by_model(model, df, ref_data_list=None):
    """
    Extracts the AUC scores for a specified model from the dataframe.

    Parameters:
    - model (str): The name of the model for which AUC scores are to be extracted.
    - df (DataFrame): The dataframe containing AUC scores along with model names.

    Returns:
    - A tuple containing:
        - A numpy array of log-transformed AUC scores for the specified model.
        - A list of data identifiers corresponding to each score (only if model is 'preferentialAttachment').
    """
    AUCs = df.loc[df["model"] == model].sort_values("data")
    data_list = AUCs["data"].to_list()

    # Check if all data identifiers are the same
    if ref_data_list is not None:
        assert all(d == ref_data_list[i] for i, d in enumerate(data_list))

    AUCs = AUCs["log_score"].values
    if model == "preferentialAttachment":
        return AUCs, data_list
    else:
        return AUCs


def cal_auc_ratio(df, n_model):

    df_plot_list = []
    for sample_id, dg in df.groupby("sampleId"):
        PA_auc_arr, data_list = auc_by_model("preferentialAttachment", dg)

        diff_arr = np.array(
            [
                auc_by_model(model, df, ref_data_list=data_list) - PA_auc_arr
                for model in model_list
            ]
        )
        diff_arr = diff_arr.reshape(-1)
        df_plot_list.append(
            pd.DataFrame(
                data={
                    "Score": diff_arr,
                    "Data": data_list * n_model,
                    "sample": sample_id,
                    "model": sum([[m] * len(data_list) for m in model_list], []),
                }
            )
        )
    df_plot = pd.concat(df_plot_list, ignore_index=True)
    return df_plot


def get_joint_df_plot(df_uniform, df_biased, n_model):
    df_plot_uniform = cal_auc_ratio(df_uniform, n_model)
    df_plot_degreeBiased = cal_auc_ratio(df_biased, n_model)

    df_plot_uniform["negativeEdgeSampler"] = ["Uniform"] * df_plot_uniform.shape[0]
    df_plot_degreeBiased["negativeEdgeSampler"] = [
        "Biased"
    ] * df_plot_degreeBiased.shape[0]
    df_plot = pd.concat([df_plot_uniform, df_plot_degreeBiased], ignore_index=True)
    df_plot.Data = pd.Categorical(df_plot.Data)
    df_plot["data_code"] = df_plot.Data.cat.codes
    return df_plot


# ========================
# Load
# ========================
df_result_auc_roc = pd.read_csv(
    auc_roc_table_file,
)

# ========================
# Preprocessing
# ========================
df_result_auc_roc["log_score"] = np.log(df_result_auc_roc["score"].values)
df_uniform = df_result_auc_roc[df_result_auc_roc["negativeEdgeSampler"] == "uniform"]
df_degreeBiased = df_result_auc_roc[
    df_result_auc_roc["negativeEdgeSampler"] == "degreeBiased"
]
model_list = list(set(df_uniform["model"]))
model_list.remove("preferentialAttachment")
n_model = len(model_list)

plot_data = get_joint_df_plot(df_uniform, df_degreeBiased, n_model)

df = (
    plot_data.query("negativeEdgeSampler == 'Uniform'")
    .groupby("Data")[["Score"]]
    .mean()
    .reset_index()
    .sort_values(by="Score")
)
df["data_code_sorted"] = np.arange(df.shape[0], dtype="int")
plot_data = pd.merge(plot_data, df.drop(columns="Score"), on="Data")
plot_data = plot_data.drop(columns="data_code").rename(
    columns={"data_code_sorted": "data_code"}
)
plot_data["negativeEdgeSampler"] = plot_data["negativeEdgeSampler"].map(
    lambda x: "Bias aligned" if x == "Biased" else x
)


score_max = plot_data["Score"].values.max()
score_min = plot_data["Score"].values.min()

# Let's focus on the "learning" methods
exclude = [
    "dcGCN",
    "dcGAT",
    "dcGraphSAGE",
    "dcGIN",
    #    "jaccardIndex",
    #    "commonNeighbors",
    #    "resourceAllocation",
    #    "localPathIndex",
    #    "localRandomWalk",
]
plot_data = plot_data[~plot_data["model"].isin(exclude)]

# %% ========================
# Plot
# ========================
# Create a scatter plot
sns.set_style("white")
sns.set(font_scale=1.4)
sns.set_style("ticks")

df = plot_data.query("Score > -1.1 and Score < 1.1").copy()
df["Score"] = np.exp(df["Score"].values)

g = sns.jointplot(
    data=df.sort_values("negativeEdgeSampler", ascending=False),
    x="data_code",
    y="Score",
    kind="scatter",
    joint_kws={"s": 23, "alpha": 0.35, "linewidth": 0},
    marginal_kws={"log_scale": True},
    hue="negativeEdgeSampler",
    palette={"Uniform": "#cdcdcd", "Bias aligned": sns.color_palette("bright")[1]},
)
# g.fig.set_figwidth(9 * 0.815)
# g.fig.set_figheight(5 * 1.16)
g.fig.set_figwidth(8 * 0.815)
g.fig.set_figheight(5 * 1.16 * 1.66 / 2)
g.ax_joint.set_xscale("linear")
g.ax_marg_x.remove()
g.ax_joint.tick_params(labelbottom=False)
g.ax_joint.set_ylabel("Ratio of AUC-ROC to\nPreferential Attachment")
g.ax_joint.set_xlabel("Graphs")
g.ax_joint.axhline(y=1, linewidth=1.5, color="black", linestyle="--")
g.ax_marg_y.axhline(y=1, linewidth=1.5, color="black", linestyle="--")
g.ax_joint.set(xlim=(0, 92), ylim=(score_min - 1e-1, score_max + 1e-1))
handles, labels = g.ax_joint.get_legend_handles_labels()
g.ax_joint.legend(
    handles[::-1],
    labels[::-1],
    frameon=False,
    loc="upper left",
    handletextpad=0.1,
    markerscale=2,
)

g.savefig(output_file, bbox_inches="tight", dpi=300)
# %%


sns.set_style("white")
sns.set(font_scale=1.4)

sns.set_style("ticks")

df = plot_data.query(
    "negativeEdgeSampler == 'Uniform' and Score > -1.1 and Score < 1.1"
).copy()
df["Score"] = np.exp(df["Score"].values)
score_max = df["Score"].values.max()
score_min = df["Score"].values.min()
g = sns.jointplot(
    data=df,
    x="data_code",
    y="Score",
    kind="scatter",
    joint_kws={"s": 23, "alpha": 0.35, "linewidth": 0},
    marginal_kws={"log_scale": True},
    hue="negativeEdgeSampler",
    palette={"Uniform": "#8d8d8d", "Bias aligned": sns.color_palette()[1]},
)
g.ax_joint.set_xscale("linear")

# Get the current locations and labels
# locs, labels = g.ax_joint.get_yticks(), g.ax_joint.get_yticklabels()
# locs = np.linspace(0.3, 1.4, int((1.4 - 0.3) / 0.1) + 2)
# Set the y-tick labels to linear
# g.ax_joint.set_yticks(locs, list(map(lambda x: f"{x:.02f}", locs)))


g.fig.set_figwidth(8 * 0.815)
g.fig.set_figheight(5 * 1.16 * 1.66 / 2)
g.ax_marg_x.remove()
g.ax_joint.set_ylabel("Ratio of AUC-ROC to\nPreferential Attachment")
g.ax_joint.set_xlabel("Graphs")
g.ax_joint.axhline(y=1, linewidth=1.5, color="black", linestyle="--")
g.ax_marg_y.axhline(y=1, linewidth=1.5, color="black", linestyle="--")

g.ax_joint.set(xlim=(0, 92), ylim=(score_min - 1e-1, score_max + 1e-1))
g.ax_joint.legend().remove()

labels = [item.get_text() for item in g.ax_joint.get_yticklabels()]

# g.ax_joint.set_yticklabels([])
# g.ax_marg_y.set_xticklabels([])
g.savefig(output_file_uniform, bbox_inches="tight", dpi=300)

# %%
# df = plot_data.copy()
# df["Score"] = np.exp(df["Score"].values)
# df = (
#    df.query("negativeEdgeSampler =='Bias aligned' ")[["Data", "Score", "model"]]
#    .groupby(["Data", "model"])
#    .mean()
# )
# np.mean(df["Score"].values < 0)
#
## %%
# df["Score"].min()
#
## %%
#
