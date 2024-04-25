# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-01-17 08:42:24
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-07-28 16:22:24
# %%
import ast
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import glob
import pathlib
import pandas as pd
from tqdm import tqdm


def load_files(dirname):
    if isinstance(dirname, str):
        input_files = list(glob.glob(dirname))
    else:
        input_files = dirname

    def get_params(filenames):
        def _get_params(filename, sep="~"):
            params = pathlib.Path(filename).stem.split("_")
            retval = {"filename": filename}
            for p in params:
                if sep not in p:
                    continue
                kv = p.split(sep)

                retval[kv[0]] = kv[1]
            return retval

        return pd.DataFrame([_get_params(filename) for filename in filenames])

    df = get_params(input_files)
    filenames = df["filename"].drop_duplicates().values
    dglist = []
    for filename in tqdm(filenames):
        dg = pd.read_csv(filename)
        dg["filename"] = filename
        dglist += [dg]
    dg = pd.concat(dglist)
    df = pd.merge(df, dg, on="filename")
    return df


def to_numeric(df, to_int, to_float):
    df = df.astype({k: float for k in to_int + to_float}, errors="ignore")
    df = df.astype({k: int for k in to_int}, errors="ignore")
    return df


if "snakemake" in sys.modules:
    input_file_list = snakemake.input["input_file_list"]
    param_file = snakemake.params["param_file"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/"
    output_file = "../data/"

# ========================
# Load
# ========================
data_table = load_files(input_file_list)
param_table = pd.read_csv(param_file)

# Append sampleId
data_table = pd.merge(
    data_table, param_table, left_on="trainTestSplit", right_on="hash", how="left"
)
data_table["sampleId"] = (
    data_table["paramValue"].apply(ast.literal_eval).apply(lambda x: x["sampleId"])
)
data_table = data_table.drop(columns=param_table.columns)

# Add the negtive sampler name used to train the model
data_table = pd.merge(
    data_table, param_table, left_on="PredictionModel", right_on="hash", how="left"
)
data_table["model"] += (
    data_table["paramValue"]
    .apply(ast.literal_eval)
    .apply(
        lambda x: "+" + x["negative_edge_sampler"]
        if "negative_edge_sampler" in x
        else ""
    )
)
data_table = data_table.drop(columns=param_table.columns)

data_table.to_csv(output_file)
