# %%
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import special

xmin = 1

xmax = 1000
x = np.arange(xmin, xmax)
alpha_list = []
aucroc_list = []
approx_aucroc_list = []
kmin = 1
for alpha in np.linspace(2.01, 5, 10):
    s = (
        np.power(x, -alpha)
        * (
            scipy.special.zeta(alpha - 1, q=x + 1)
            / scipy.special.zeta(alpha - 1, q=xmin)
        )
        / scipy.special.zeta(alpha, q=xmin)
    )
    aucroc_assortative = np.sum(s)
    aucroc = aucroc_assortative**2

    aucroc_approx_disc = special.zeta(2 * alpha - 1, kmin + 1) / (
        special.zeta(alpha, kmin) * special.zeta(alpha - 1, kmin)
    )

    alpha_list.append(alpha)
    aucroc_list.append(aucroc)
    approx_aucroc_list.append(aucroc_assortative)

alphas = np.array(alpha_list)
aucrocs = np.array(aucroc_list)
approx_aucrocs = np.array(approx_aucroc_list)


import pandas as pd

df = pd.DataFrame({"alpha": alphas, "aucroc": aucrocs, "approx_aucroc": approx_aucrocs})

sns.lineplot(data=df, x="alpha", y="aucroc", label="exact")
sns.lineplot(data=df, x="alpha", y="approx_aucroc", label="approx")

# %%
