# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-05-05 08:44:53
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-07-28 16:18:52
"""Loads and preprocesses tables of AUC-ROC scores, rankings, and
network statistics, computes the quantiles of the scores among models
for each dataset/metric combination, and creates a new table containing
the quantiles of link prediction performance across different algorithms.
"""
# %%
import numpy as np
import pandas as pd
import sys
import ast

if "snakemake" in sys.modules:
    auc_roc_table_file = snakemake.input["auc_roc_table_file"]
    ranking_table_file = snakemake.input["ranking_table_file"]
    param_file = snakemake.params["param_file"]
    net_stat_file = snakemake.input["net_stat_file"]
    output_file = snakemake.output["output_file"]
else:
    auc_roc_table_file = "../../data/derived/results/result_auc_roc.csv"
    ranking_table_file = "../../data/derived/results/result_ranking.csv"
    param_file = "../../data/parameter_table.csv"
    output_file = "../data/derived/results/result_quantile_ranking.csv"

# ========================
# Load
# ========================

param_table = pd.read_csv(param_file)

aucroc_table = pd.read_csv(
    auc_roc_table_file,
    # usecols=["sampleId", "data", "score", "negativeEdgeSampler", "model"],
).drop_duplicates()

ranking_table = pd.read_csv(
    ranking_table_file,
).drop_duplicates()


## Append meta data
# aucroc_table = pd.merge(
#    aucroc_table, param_table, left_on="trainTestSplit", right_on="hash", how="left"
# )
# aucroc_table["sampleId"] = (
#    aucroc_table["paramValue"].apply(ast.literal_eval).apply(lambda x: x["sampleId"])
# )
#
# ranking_table = pd.merge(
#    ranking_table, param_table, left_on="trainTestSplit", right_on="hash", how="left"
# )
# ranking_table["sampleId"] = (
#    ranking_table["paramValue"].apply(ast.literal_eval).apply(lambda x: x["sampleId"])
# )
# %%
# ========================
# Preprocessing
# ========================
#
# Compute the ranking
#
results = []
for (data, negSampler), df in aucroc_table.groupby(["data", "negativeEdgeSampler"]):
    df = (
        df.groupby(["data", "negativeEdgeSampler", "model"])
        .mean(numeric_only=True)
        .reset_index()
    )
    df = df.sort_values(by="score")
    df["quantile"] = np.arange(df.shape[0]) / (df.shape[0] - 1)
    df["rank"] = df.shape[0] - np.arange(df.shape[0])
    df["metric"] = "AUCROC+" + df["negativeEdgeSampler"]
    results.append(df)

for (data, metric), df in ranking_table.groupby(["data", "metric"]):
    df = df.groupby(["model", "data", "metric"]).mean(numeric_only=True).reset_index()
    df = df.sort_values(by="score")
    df["quantile"] = np.arange(df.shape[0]) / (df.shape[0] - 1)
    df["rank"] = df.shape[0] - np.arange(df.shape[0])
    results.append(df)
results = pd.concat(results)

# data_table = results.pivot(
#    columns="metric",
#    values="quantile",
#    index=[
#        "data",
#        # "sampleId",
#        "model",
#    ],
# ).reset_index()

results.to_csv(output_file, sep=",", index=False)

# %%
