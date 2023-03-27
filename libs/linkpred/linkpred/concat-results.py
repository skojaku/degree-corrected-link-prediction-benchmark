# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-01-17 08:42:24
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-01-17 09:01:00
# %%
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
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/"
    output_file = "../data/"

# ========================
# Load
# ========================
data_table = load_files(input_file_list)
data_table.to_csv(output_file)
