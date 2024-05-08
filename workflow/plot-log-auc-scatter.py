# %%
import pandas as pd
import seaborn as sns
import numpy as np
import sys


if "snakemake" in sys.modules:
    auc_roc_table_file = snakemake.input["auc_roc_table_file"]
    output_file = snakemake.output["output_file"]
else:
    auc_roc_table_file = "../data/derived/results/result_auc_roc.csv"
    output_file = "../figs/log_auc_scatter_plot.pdf"


def auc_by_model(model, df):
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
            [auc_by_model(model, df) - PA_auc_arr for model in model_list]
        )
        diff_arr = diff_arr.reshape(-1)
        df_plot_list.append(
            pd.DataFrame(
                data={
                    "Score": diff_arr,
                    "Data": data_list * n_model,
                    "sample": sample_id,
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

score_max = plot_data["Score"].values.max()
score_min = plot_data["Score"].values.min()

# %% ========================
# Plot
# ========================
# Create a scatter plot
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

g = sns.jointplot(
    data=plot_data,
    x="data_code",
    y="Score",
    kind="scatter",
    joint_kws={"s": 23, "alpha": 0.35, "linewidth": 0},
    hue="negativeEdgeSampler",
)
g.fig.set_figwidth(8)
g.fig.set_figheight(5)
g.ax_marg_x.remove()
g.ax_joint.tick_params(labelbottom=False)
g.ax_joint.set(
    xlabel="Data",
    ylabel="Relative performance to Preferential Attachment",
)
g.ax_joint.axhline(y=0, linewidth=1, color="black", linestyle="--")
g.ax_joint.set(xlim=(0, 92), ylim=(score_min - 0.1, score_max + 0.1))
g.ax_joint.legend(frameon=False, loc="lower left")
g.savefig(output_file, bbox_inches="tight", dpi=300)

# %%
