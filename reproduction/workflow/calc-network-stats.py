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
from typing import Optional

if "snakemake" in sys.modules:
    input_files = snakemake.input["input_files"]
    output_file = snakemake.output["output_file"]
else:
    from glob import glob

    input_files = list(glob("../data/preprocessed/*/edge_table.csv"))
    output_file = "../data/stats/network-stats.csv"


def estimate_avg_path_length(
    g: igraph.Graph, num_samples: int = 1000, seed: Optional[int] = None
) -> float:
    """
    Estimate average path length in a graph using random sampling.

    Parameters:
    -----------
    g : igraph.Graph
        Input graph
    num_samples : int
        Number of random node pairs to sample
    seed : Optional[int]
        Random seed for reproducibility

    Returns:
    --------
    tuple[float, float]
        (estimated_avg_path_length, standard_error)
    """
    if seed is not None:
        np.random.seed(seed)

    # Get number of vertices
    n = g.vcount()

    # Store path lengths
    path_lengths = []

    # Sample random pairs and compute shortest paths
    for _ in range(num_samples):
        # Sample two different nodes
        v1, v2 = np.random.choice(range(n), 2, replace=False)

        # Compute shortest path length
        path_length = g.distances(v1, v2)[0]

        # Infinite path length means nodes are in different components
        if path_length != float("inf"):
            path_lengths.append(path_length)

    # Compute statistics
    avg_path_length = np.mean(path_lengths)

    return avg_path_length


results = []
pbar = tqdm(input_files)
for filename in pbar:
    dataname = filename.split("/")[-2]
    edge_table = pd.read_csv(filename)
    r, c = tuple(edge_table[["src", "trg"]].values.T)
    n_nodes = np.maximum(np.max(r), np.max(c)) + 1
    A = utils.edgeList2adjacencyMatrix(r, c, n_nodes)
    deg = np.array(A.sum(axis=1)).reshape(-1)
    n_edges = int(np.sum(deg) / 2)

    pbar.set_postfix(dataname=dataname, n_nodes=n_nodes, n_edges=n_edges)

    g = igraph.Graph(list(zip(r, c)), directed=False)
    global_transitivity = g.transitivity_undirected()
    local_transitivity = np.array(g.transitivity_local_undirected())
    assortativity = g.assortativity_degree(directed=False)
    # average_path_length = estimate_avg_path_length(g)
    average_path_length = g.average_path_length()

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
    #    try:
    #        res = powerlaw.Fit(deg)
    #        alpha = res.power_law.alpha
    #        xmin = res.power_law.xmin
    #    except:
    #        alpha = np.nan
    #        xmin = np.nan

    results += [
        {
            "network": dataname,
            "n_nodes": n_nodes,
            "n_edges": len(r),
            "edge_density": density,
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
            "averagePathLength": average_path_length,
            "degreeHeterogeneity": Hm,
            "lognorm_mu": lognorm_mu,
            "lognorm_sigma": lognorm_sigma,
        }
    ]

results = pd.DataFrame(results)
results.to_csv(output_file, index=False)

# %%
g.average_path_length()
# %%
distances
# %%
