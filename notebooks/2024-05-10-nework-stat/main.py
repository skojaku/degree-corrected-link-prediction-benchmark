# %%
import numpy as np
import pandas as pd

data_table = pd.read_csv("../../data/stats/network-stats.csv")

# %%
df = data_table[data_table["network"].apply(lambda x: "ogbl" in x)]

df = df[["network", "n_nodes", "n_edges", "maxDegree", "degreeVariance", "degreeAssortativity", "lognorm_sigma"]].rename(columns = {
    "network": "Network",
    "n_nodes": "Nodes",
    "n_edges": "Edges",
    "maxDegree": "Max. Degree",
    "degreeVariance": "Variance",
    "degreeAssortativity": "Assortativity",
    "lognorm_sigma": "Heterogeneity",
})
df

# %%
data_table.shape
#.to_latex(index=False)
# %%
df.columns
#Network &  Nodes &  Edges &  Max. Degree &  Variance &  Assortativity & Heterogeneity

# %%
import seaborn as sns

sns.heatmap(
    data_table.query("n_nodes > 3000").drop(columns=["network"]).corr("spearman"),
    cmap="coolwarm",
)

# %%
data_table.query("n_nodes > 3").sort_values("degreeVariance", ascending=False).head(10)
# %%
data_table.sort_values("degreeVariance_normalized", ascending=False).head(10)

# %%
data_table.query("n_nodes > 3").shape

# %%
