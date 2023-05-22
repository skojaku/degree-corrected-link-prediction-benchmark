# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-05-22 13:55:10
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-22 15:02:26
# %%
import pathlib
import numpy as np
import pandas as pd
from scipy import sparse
import sys
from tqdm import tqdm
import glob

if "snakemake" in sys.modules:
    net_files = snakemake.input["net_files"]
    edge_table_files = snakemake.input["edge_table_files"]
    output_file = snakemake.output["output_file"]
else:
    net_files = glob.glob(
        "../data/derived/datasets/*/train-net_testEdgeFraction~0.5_sampleId~*.npz"
    )
    edge_table_files = glob.glob(
        "../data/derived/datasets/*/testEdgeTable_testEdgeFraction~0.5_sampleId~*.csv"
    )
    output_file = "../data/"


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
    return df


def to_numeric(df, to_int, to_float):
    df = df.astype({k: float for k in to_int + to_float}, errors="ignore")
    df = df.astype({k: int for k in to_int}, errors="ignore")
    return df


def calc_common_neighbor_edge_coverage(edge_table_file, net_file, n_steps, **params):
    # ========================
    # Load
    # ========================
    test_edge_table = pd.read_csv(edge_table_file)
    train_net = sparse.load_npz(net_file)

    # ========================
    # Preprocess
    # ===========================
    rows, cols = tuple(
        test_edge_table.query("isPositiveEdge == 1")[["src", "trg"]].values.T
    )
    rows, cols = np.concatenate([rows, cols]), np.concatenate([cols, rows])
    test_net = sparse.csr_matrix(
        (np.ones_like(rows), (rows, cols)), shape=train_net.shape
    )
    test_net.data = test_net.data * 0 + 1

    P = train_net @ train_net
    it = 3
    while it <= n_steps:
        P = P + P @ train_net
    src, trg, _ = sparse.find(P)
    connected = np.array(train_net[(src, trg)]).reshape(-1) > 0
    connected = connected | (src == trg)
    src, trg = src[~connected], trg[~connected]
    Ypred = sparse.csr_matrix((np.ones_like(src), (src, trg)), shape=train_net.shape)
    hit = (Ypred.multiply(test_net)).sum()
    coverage = hit / test_net.sum()
    density = len(src) / np.prod(test_net.shape)
    return coverage, density


# ========================
# Load
# ========================

net_table = load_files(net_files)
test_edge_table = load_files(edge_table_files)


# Get the data out
net_table["data"] = net_table["filename"].str.split("/").apply(lambda x: x[-2])
test_edge_table["data"] = (
    test_edge_table["filename"].str.split("/").apply(lambda x: x[-2])
)

results = []
for i, row in tqdm(
    pd.merge(
        net_table.rename(columns={"filename": "net_file"}),
        test_edge_table.rename(columns={"filename": "edge_table_file"}),
        on=["data", "sampleId", "testEdgeFraction"],
    ).iterrows(),
    total=net_table.shape[0],
):
    coverage, density = calc_common_neighbor_edge_coverage(n_steps=3, **row)
    row["common_neighbor_coverage"] = coverage
    row["density"] = density
    results.append(row)
df = pd.DataFrame(results)
df.to_csv(output_file, index=False)

# %%
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# net_stat_file = "../data/derived/networks/network-stats.csv"
# net_stat_table = pd.read_csv(net_stat_file).rename(columns={"network": "data"})
#
# dg = df.groupby("data").mean()["common_neighbor_coverage"].reset_index()
# dg = pd.merge(dg, net_stat_table, on="data")
# dg["data"] = dg.apply(lambda x: "%s (%d)" % (x["data"], x["n_nodes"]), axis=1)
# dg
## %%
#
# sns.set_style("white")
# sns.set(font_scale=1.2)
# sns.set_style("ticks")
# fig, ax = plt.subplots(figsize=(5, 30))
#
# sns.barplot(
#    data=dg,
#    y="data",
#    x="common_neighbor_coverage",
#    order=dg.sort_values(by="common_neighbor_coverage")["data"],
#    ax=ax,
# )
# ax.axvline(0.5, ls=":", color="k")
## %%
#
