# %%
import numpy as np
import pandas as pd

data_table = pd.read_csv("../../data/stats/network-stats.csv")

# %%
data_table.describe()

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
