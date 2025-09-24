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
import igraph
from tqdm import tqdm
import powerlaw

if "snakemake" in sys.modules:
    input_files = snakemake.input["input_files"]
    output_file = snakemake.output["output_file"]
else:
    from glob import glob

    input_files = list(glob("../data/preprocessed/*/edge_table.csv"))
    output_file = "../data/stats/network-stats.csv"

results = []


# %%
for filename in tqdm(input_files):
    dataname = filename.split("/")[-2]
    edge_table = pd.read_csv(filename)
    r, c = tuple(edge_table[["src", "trg"]].values.T)
    n_nodes = np.maximum(np.max(r), np.max(c)) + 1
    A = utils.edgeList2adjacencyMatrix(r, c, n_nodes)
    deg = np.array(A.sum(axis=1)).reshape(-1)
    n_edges = int(np.sum(deg) / 2)

    g = igraph.Graph(list(zip(r, c)), directed=False)
    global_transitivity = g.transitivity_undirected()
    local_transitivity = np.array(g.transitivity_local_undirected())
    assortativity = g.assortativity_degree(directed=False)

    _, p = np.unique(deg, return_counts=True)
    p = p / np.sum(p)
    h = np.sqrt(np.sum((1 - p) ** 2) / n_nodes)
    Hm = h / (np.sqrt(1 - 3 / n_nodes))

    deg_variance = np.var(deg)
    density = np.sum(deg) / (n_nodes * (n_nodes - 1))
    normalized_deg_variance = (
        (n_nodes - 1) * deg_variance / (n_nodes * n_edges * (1 - density))
    )

    # Log normal distribution
    lognorm_sigma, _, lognorm_mu = stats.lognorm.fit(deg, floc=0, method="MM")

    # Power law distribution?
    try:
        res = powerlaw.Fit(deg)
        alpha = res.power_law.alpha
        xmin = res.power_law.xmin
    except:
        alpha = np.nan
        xmin = np.nan

    results += [
        {
            "network": dataname,
            "n_nodes": n_nodes,
            "n_edges": len(r),
            "averageDegree": np.mean(deg),
            "maxDegree": np.max(deg),
            "degreeKurtosis": stats.kurtosis(deg),
            "degreeSkewness": stats.skew(deg),
            "degreeVariance": deg_variance,
            "degreeVariance_normalized": normalized_deg_variance,
            "degreeAssortativity": assortativity,
            "globalTransitivity": global_transitivity,
            "localTransitivity": np.mean(
                local_transitivity[~pd.isna(local_transitivity)]
            ),
            "degreeHeterogeneity": Hm,
            "lognorm_mu": lognorm_mu,
            "lognorm_sigma": lognorm_sigma,
        }
    ]

results = pd.DataFrame(results)
results.to_csv(output_file, index=False)

# %%
results.query("network == 'ogbl-collab'")
# %%
