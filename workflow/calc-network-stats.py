# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-05-03 08:53:20
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-03 09:03:57
# %%
import numpy as np
import pandas as pd
import sys
import utils
from scipy import stats

if "snakemake" in sys.modules:
    input_files = snakemake.input["input_files"]
    output_file = snakemake.output["output_file"]
else:
    from glob import glob

    input_files = list(glob("../data/derived/networks/preprocessed/*/edge_table.csv"))
    output_file = "../data/"

results = []
for filename in input_files:
    dataname = filename.split("/")[-2]
    edge_table = pd.read_csv(filename)
    r, c = tuple(edge_table[["src", "trg"]].values.T)
    n_nodes = np.maximum(np.max(r), np.max(c)) + 1
    A = utils.edgeList2adjacencyMatrix(r, c, n_nodes)
    deg = np.array(A.sum(axis=1)).reshape(-1)
    results += [
        {
            "network": dataname,
            "n_nodes": n_nodes,
            "n_edges": len(r),
            "maxDegree": np.max(deg),
            "degreeKurtosis": stats.kurtosis(deg),
            "degreeAssortativity": stats.pearsonr(deg[r], deg[c])[0],
        }
    ]
results = pd.DataFrame(results)
results.to_csv(output_file, index=False)
