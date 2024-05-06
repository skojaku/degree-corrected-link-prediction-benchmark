import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys


if "snakemake" in sys.modules:
    auc_roc_table_file = snakemake.input["auc_roc_table_file"]
    output_file = snakemake.output["output_file"]
else:
    auc_roc_table_file = "../data/derived/results/result_auc_roc.csv"
    output_file = "../figs/log_auc_scatter_plot.pdf"

# ========================
# Load
# ========================
aucroc_table = pd.read_csv(
    auc_roc_table_file,
) 

# ========================
# Scatter plot
# ========================

def auc_by_model(model, df):
    AUCs = df.loc[df["model"] == model].sort_values("data")
    data_list = AUCs["data"].to_list()
    AUCs = AUCs["log_score"].values
    if model == "preferentialAttachment":
        return AUCs, data_list
    else:
        return AUCs

def cal_auc_ratio(df, n_model):
    PA_auc_arr, data_list = auc_by_model("preferentialAttachment", df)
    
    diff_arr = np.array([auc_by_model(model, df)-PA_auc_arr for model in model_list])
    diff_arr = diff_arr.reshape(-1)
    df_plot = pd.DataFrame(data={"Score": diff_arr, "Data": data_list * n_model})
    return df_plot

def get_joint_df_plot(df_uniform, df_biased, n_model):
    df_plot_uniform = cal_auc_ratio(df_uniform, n_model)
    df_plot_degreeBiased = cal_auc_ratio(df_biased, n_model)
    
    df_plot_uniform["negativeEdgeSampler"] = ["Uniform"] * df_plot_uniform.shape[0]
    df_plot_degreeBiased["negativeEdgeSampler"] = ["Biased"] * df_plot_degreeBiased.shape[0]
    df_plot = pd.concat([df_plot_uniform, df_plot_degreeBiased], ignore_index = True)
    df_plot.Data = pd.Categorical(df_plot.Data)
    df_plot['data_code'] = df_plot.Data.cat.codes
    return df_plot

def scatter_plot(df_plot):
    score_max = df_plot["Score"].values.max()
    score_min = df_plot["Score"].values.min()
    
    # Create a scatter plot
    
    g = sns.jointplot(data = df_plot, x="data_code", y="Score", kind='scatter', joint_kws={"s": 23, "alpha":0.35, "linewidth":0}, hue="negativeEdgeSampler")
    g.fig.set_figwidth(8)
    g.fig.set_figheight(5)
    g.ax_marg_x.remove()
    g.ax_joint.tick_params(labelbottom=False)
    g.ax_joint.set(xlabel='Data', ylabel="Log returns of AUC score")
    g.ax_joint.axhline(y=0, linewidth=1, color='black', linestyle="--")
    g.ax_joint.set(xlim=(0, 92), ylim=(score_min-0.1, score_max+0.1))
    g.ax_joint.legend(frameon=False, loc="lower left")
    g.savefig(output_file, bbox_inches="tight", dpi=300)
    
    

df_result_auc_roc = pd.read_csv("result_auc_roc.csv")
df_result_auc_roc["log_score"] = np.log(df_result_auc_roc["score"].values)
df_uniform = df_result_auc_roc[df_result_auc_roc["negativeEdgeSampler"]=="uniform"]
df_degreeBiased = df_result_auc_roc[df_result_auc_roc["negativeEdgeSampler"]=="degreeBiased"]
model_list = list(set(df_uniform["model"]))
model_list.remove("preferentialAttachment")
n_model = len(model_list)
df_plot = get_joint_df_plot(df_uniform, df_degreeBiased, n_model)
get_fig = scatter_plot(df_plot)