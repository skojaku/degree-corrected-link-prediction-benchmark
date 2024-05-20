# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


data_table = pd.read_csv("../../data/derived/results/result_auc_roc.csv")
data_table

# %%
data_table.query("negativeEdgeSampler == 'degreeBiased'")[["model", "score"]].groupby(["model"]).describe().sort_values(by=("score", "mean"), ascending=False)
# %%

data_table["model"].unique().shape[0] - 4

# %%
