# %%
import numpy as np
from scipy import sparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

from gnn_tools.LinkPredictionDataset import LinkPredictionDataset


# %% =======================================
# Generate the benchmark
# =======================================
network_name = "opsahl-openflights"
# network_name = "airport-rach"
# network_name = "dblp-cite"
fraction = 0.2  # Fraction of edges to use for testing

network_file = f"../../data/networks/preprocessed/{network_name}/edge_table.csv"

edge_table = pd.read_csv(network_file)

# Construct the network
src, trg = edge_table["src"].values, edge_table["trg"].values
n_nodes = int(np.maximum(np.max(src), np.max(trg)) + 1)
net = sparse.csr_matrix((np.ones_like(src), (src, trg)), shape=(n_nodes, n_nodes))
net = net+ net.T
net.data = np.ones_like(net.data)

# Generate the link prediction benchmark
model = LinkPredictionDataset(
    testEdgeFraction=fraction,
    negative_edge_sampler="degreeBiased",
)
model.fit(net)
train_val_net, test_edges = model.transform()

# Validation
model_val = LinkPredictionDataset(
    testEdgeFraction=0.5 * fraction,
    negative_edge_sampler="uniform",
)
model_val.fit(train_val_net)
train_net, val_edges = model_val.transform()
src_val, trg_val, y_val = (
    val_edges["src"],
    val_edges["trg"],
    val_edges["isPositiveEdge"],
)
deg_val = np.array(train_net.sum(axis=1)).flatten()
p_val = deg_val / np.sum(deg_val)
# %% =======================================
# Training
# ==========================================
dim = 128
from sklearn.linear_model import LogisticRegression

embedding_model_list = [
    "leigenmap",
    # "deepwalk",
    # "node2vec",
    # "modspec",
    "GraphSAGE",
    "GAT",
    "dcGraphSAGE",
    "dcGAT",
]
Semb = {}
for model_name in embedding_model_list:
    emb = embedding_models[model_name](
        train_net,
        dim=dim,
        feature_dim=dim,
        epochs=500,
    )
    S = emb @ emb.T

    Semb[model_name] = S

# %%
topology_model_list = [
    "commonNeighbors",
    "jaccardIndex",
    "adamicAdar",
    "resourceAllocation",
]

Stop = {}
src, trg = np.triu_indices(train_net.shape[0], 1)
for model_name in topology_model_list:
    score = topology_models[model_name](train_net, src, trg)
    net_predicted = np.zeros((train_net.shape[0], train_net.shape[1]))
    net_predicted[src, trg] = score
    net_predicted[trg, src] = score
    Stop[model_name] = net_predicted

# %% =======================================
# Compute the AUC-ROC
# ==========================================
from sklearn.metrics import roc_auc_score

src, trg, y = test_edges["src"], test_edges["trg"], test_edges["isPositiveEdge"]

results = []
for model_name in embedding_model_list:
    score = Semb[model_name][(src, trg)]
    auc_roc = roc_auc_score(y, score)
    results.append(
        {
            "model": model_name,
            "score": auc_roc,
            "type": "embedding",
            "scoreType": "aucroc",
        }
    )
    print(f"{model_name}: {auc_roc}")

for model_name in topology_model_list:
    score = Stop[model_name][(src, trg)]
    auc_roc = roc_auc_score(y, score)
    results.append(
        {
            "model": model_name,
            "score": auc_roc,
            "type": "topology",
            "scoreType": "aucroc",
        }
    )
    print(f"{model_name}: {auc_roc}")

# %% =======================================
# Compute the precision & recall
# ==========================================
topk = 10
test_net = sparse.csr_matrix(
    (np.ones(2 * len(src)), (np.concatenate([src, trg]), np.concatenate([trg, src]))),
    shape=(train_net.shape[0], train_net.shape[0]),
)
deg_test = np.array(test_net.sum(axis=1)).flatten()
focal_nodes = np.where(deg_test >= 1)[0]
for model_name in embedding_model_list:
    S = Semb[model_name]

    #    clf = LogisticRegression(random_state=0).fit(
    #        (
    #            np.vstack(
    #                [
    #                    S[(src_val, trg_val)],
    #                    np.log(p_val[src_val] * p_val[trg_val]),
    #                ]
    #            ).T
    #        ),
    #        y_val,
    #    )
    clf = LogisticRegression(random_state=0, penalty=None).fit(
        S[(src_val, trg_val)].reshape((-1, 1)),
        y_val,
    )
    w = clf.coef_[0, 0]
    # w1 = clf.coef_[0, 1]
    b = clf.intercept_
    S = np.outer(p_val, p_val) * 1.0 / (1.0 + np.exp(-(w * S + b)))
    # S = S + np.add.outer(np.log(p_val), np.log(p_val))
    # print(w, b)

    candidates = np.argsort(-S[focal_nodes, :], axis=1)[:, :topk]
    src_test = (focal_nodes.reshape((-1, 1)) @ np.ones((1, topk))).flatten().astype(int)
    src_test, candidates = src_test.reshape(-1), candidates.reshape(-1)
    pred_net = sparse.csr_matrix(
        (np.ones(len(src_test)), (src_test, candidates)), shape=test_net.shape
    )
    prec = np.sum(
        test_net[focal_nodes].multiply(pred_net[focal_nodes]).sum(axis=1).A1
    ) / np.sum(pred_net[focal_nodes].sum(axis=1).A1)
    rec = np.sum(
        test_net[focal_nodes].multiply(pred_net[focal_nodes]).sum(axis=1).A1
    ) / np.sum(test_net[focal_nodes].sum(axis=1).A1)
    print(model_name, prec, rec)
    results.append(
        {
            "model": model_name,
            "score": prec,
            "type": "embedding",
            "scoreType": "precision",
        }
    )
    results.append(
        {
            "model": model_name,
            "score": rec,
            "type": "embedding",
            "scoreType": "recall",
        }
    )
# %%
for model_name in topology_model_list:
    S = Stop[model_name]
    candidates = np.argsort(-S[focal_nodes, :], axis=1)[:, :topk]
    src_test = (focal_nodes.reshape((-1, 1)) @ np.ones((1, topk))).flatten().astype(int)
    src_test, candidates = src_test.reshape(-1), candidates.reshape(-1)
    pred_net = sparse.csr_matrix(
        (np.ones(len(src_test)), (src_test, candidates)), shape=test_net.shape
    )
    prec = np.sum(
        test_net[focal_nodes].multiply(pred_net[focal_nodes]).sum(axis=1).A1
    ) / np.sum(pred_net[focal_nodes].sum(axis=1).A1)
    rec = np.sum(
        test_net[focal_nodes].multiply(pred_net[focal_nodes]).sum(axis=1).A1
    ) / np.sum(test_net[focal_nodes].sum(axis=1).A1)
    # print(test_net.multiply(pred_net).sum(axis=1).A1) / (test_net.sum(axis=1).A1)
    print(model_name, prec, rec)
    results.append(
        {
            "model": model_name,
            "score": prec,
            "type": "topology",
            "scoreType": "precision",
        }
    )
    results.append(
        {
            "model": model_name,
            "score": rec,
            "type": "topology",
            "scoreType": "recall",
        }
    )

# %% Post-process
plot_data = pd.DataFrame(results)


# Rank the methods based on score. Group by "scoreType"


plot_data = plot_data.sort_values(by="score", ascending=False)
plot_data_list = []
for score_type, df in plot_data.groupby("scoreType"):
    df["rank"] = df["score"].rank(method="dense", ascending=False)
    plot_data_list.append(df)
plot_data = pd.concat(plot_data_list)

# Convert to wide format and join by "model" and "scoreType"
plot_data = plot_data.pivot_table(
    index=["model"], columns="scoreType", values="rank"
).reset_index()
plot_data
# %%
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(5, 5))

ax = sns.scatterplot(
    data=plot_data,
    x="aucroc",
    y="precision",
    hue="model",
    palette="tab20",
    ax=ax,
)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderpad=0)
# Draw a diagonal line from the bottom left to the top right of the plot space
ax.plot([0, 1], [0, 1], transform=ax.transAxes, color="gray", linewidth=1.5, ls="--")

# fig.savefig(output_file, bbox_extra_artists=(lgd,), bbox_inches="tight", dpi=300)
# %%
# Create a plot with two rankings of methods by "aucroc" and "precision"
fig, ax = plt.subplots(figsize=(3, 5))

# Extract rankings for aucroc and precision
aucroc_ranking = plot_data.sort_values(by="aucroc", ascending=True)
precision_ranking = plot_data.sort_values(by="precision", ascending=True)

# Assign new rank positions for plotting
aucroc_ranking["plot_rank"] = range(len(aucroc_ranking))
precision_ranking["plot_rank"] = range(len(precision_ranking))

# Merge the two rankings on the model name to align them side by side
merged_rankings = pd.merge(
    aucroc_ranking, precision_ranking, on="model", suffixes=("_aucroc", "_precision")
)

color_list = sns.color_palette("tab20").as_hex()
model_list = embedding_model_list + topology_model_list
model2color = {model: color for model, color in zip(model_list, color_list)}

for i, row in merged_rankings.iterrows():
    ax.scatter(
        x=[0],
        y=[row["plot_rank_aucroc"]],
        color=model2color[row["model"]],
    )
    ax.scatter(
        x=[1],
        y=[row["plot_rank_precision"]],
        color=model2color[row["model"]],
        label=row["model"],
    )

    ax.plot(
        [0, 1],
        [row["plot_rank_aucroc"], row["plot_rank_precision"]],
        "gray",
        zorder=0,
    )
ax.legend(title="Models", bbox_to_anchor=(1.05, 1), loc=2, borderpad=0)
# Invert the y-axis for the ranking plot
ax.invert_yaxis()
ax.axis("off")
# %%


# %%
