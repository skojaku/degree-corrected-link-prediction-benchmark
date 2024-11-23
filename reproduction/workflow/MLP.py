# %%
import torch
import numpy as np
from tqdm import tqdm
from scipy import sparse
from sklearn.metrics import roc_auc_score
from gnn_tools.LinkPredictionDataset import NegativeEdgeSampler
from NetworkTopologyPredictionModels import (
    resourceAllocation,
    commonNeighbors,
    jaccardIndex,
    adamicAdar,
    localRandomWalk,
)
from sklearn.model_selection import train_test_split


class MLP(torch.nn.Module):
    def __init__(
        self,
        with_degree=False,
        hidden_layers=[32, 16],
        dropout_rate=0.2,
        activation="relu",
    ):
        super(MLP, self).__init__()

        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.activation = activation

        if activation == "relu":
            activation = torch.nn.ReLU
        elif activation == "leaky_relu":
            activation = torch.nn.LeakyReLU
        elif activation == "sigmoid":
            activation = torch.nn.Sigmoid

        # Input features: resource allocation, common neighbors, jaccard index,
        # adamic adar, local random walk, and optionally degree
        input_dim = 4 if not with_degree else 4

        # Define layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.extend(
                [
                    torch.nn.Linear(prev_dim, hidden_dim),
                    activation(),
                    torch.nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = hidden_dim

        layers.extend([torch.nn.Linear(prev_dim, 1), torch.nn.Sigmoid()])

        self.layers = torch.nn.Sequential(*layers)
        self.with_degree = with_degree
        self.mean = None
        self.std = None

    def forward(self, x):
        # x shape: [batch_size, n_features]
        # Returns predictions in range [0,1]
        return self.layers(x).squeeze()

    def forward_edges(self, network, src, trg):
        # x shape: [batch_size, n_features]
        # Returns predictions in range [0,1]

        x = compute_network_stats(
            network,
            src,
            trg,
            with_degree=self.with_degree,
            std=self.std,
            mean=self.mean,
        )
        x = torch.FloatTensor(x).to(self.layers[0].weight.device)

        return self.layers(x).squeeze()

    def predict_proba(self, x):
        # Convenience method for getting probabilities
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def fit_scaler(self, X):
        self.mean = X.mean(axis=0).detach().cpu().numpy()
        self.std = X.std(axis=0).detach().cpu().numpy()

    def get_rescaled_features(self, X):
        return (X - torch.tensor(self.mean).to(X.device)) / torch.tensor(self.std).to(
            X.device
        )


def save_model(self, filepath):
    """Save the trained model and scaling parameters to disk.

    Args:
        filepath (str): Path where model should be saved
    """
    save_dict = {
        "model_state_dict": self.state_dict(),
        "mean": self.mean,
        "std": self.std,
        "with_degree": self.with_degree,
        "hidden_layers": self.hidden_layers,
        "dropout_rate": self.dropout_rate,
        "activation": self.activation,
    }
    torch.save(save_dict, filepath)


def load_model(filepath, device="cpu"):
    """Load a trained model from disk.

    Args:
        filepath (str): Path to saved model file
        device (str): Device to load model to ('cpu' or 'cuda')

    Returns:
        MLP: Loaded model instance
    """
    checkpoint = torch.load(filepath, map_location=device)

    # Create new model instance
    model = MLP(
        with_degree=checkpoint["with_degree"],
        hidden_layers=checkpoint["hidden_layers"],
        dropout_rate=checkpoint["dropout_rate"],
        activation=checkpoint["activation"],
    )

    # Load model parameters and scaling
    model.load_state_dict(checkpoint["model_state_dict"])
    model.mean = checkpoint["mean"]
    model.std = checkpoint["std"]

    model.to(device)
    return model


def compute_network_stats(network, src, trg, with_degree=False, std=None, mean=None):
    """Compute network statistics used as features for prediction.

    Args:
        network: scipy sparse matrix representing the network
        src: source node indices (optional)
        trg: target node indices (optional)
        maxk: number of top predictions to return (optional)

    Returns:
        Array of network statistics for each node pair
    """

    # Compute each network statistic
    # Each score is a numpy array of shape (len(src),)
    ra = resourceAllocation(network, src, trg, maxk=None)
    # cn = commonNeighbors(network, src, trg, maxk=None)
    ji = jaccardIndex(network, src, trg, maxk=None)
    aa = adamicAdar(network, src, trg, maxk=None)
    lrw = localRandomWalk(network, src, trg, maxk=None)

    if with_degree:
        # Add degree product if enabled
        deg = network.sum(axis=1).A1
        deg_prod = deg[src] * deg[trg]
        features = [
            # ra,
            # ji,
            # aa,
            # lrw,
            np.maximum(deg[trg], deg[src]),
            np.minimum(deg[trg], deg[src]),
            deg_prod,
            deg[src] + deg[trg],
        ]
    else:
        features = [ra, ji, aa, lrw]

    X = np.column_stack(features)
    if std is not None and mean is not None:
        X = (X - mean) / std
    return X


def train_mlp_with_early_stopping(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=100,
    batch_size=1024,
    lr=0.005,
    device="cpu",
    patience=30,
    min_delta=0.001,
):
    """Train MLP model for link prediction with early stopping.

    Args:
        model: MLP model instance
        network: scipy sparse adjacency matrix
        train_pos_edges: positive training edges as (src, trg) array
        train_neg_edges: negative training edges as (src, trg) array
        val_pos_edges: positive validation edges (optional)
        val_neg_edges: negative validation edges (optional)
        epochs: maximum number of training epochs
        batch_size: training batch size
        lr: learning rate
        device: torch device to use
        patience: number of epochs to wait for improvement before early stopping
        min_delta: minimum change in validation AUC to qualify as an improvement

    Returns:
        Trained model and best validation AUC score if validation data provided
    """

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    # Fit scaler
    model.fit_scaler(X_train)
    X_train_rescaled = model.get_rescaled_features(X_train)
    X_val_rescaled = model.get_rescaled_features(X_val)

    # Prepare training data
    from sklearn.model_selection import train_test_split

    # Early stopping variables
    best_val_auc = 0
    best_model = None
    epochs_no_improve = 0

    # Training loop

    model.train()
    pbar = tqdm(range(epochs), desc="Training")
    val_x = X_val
    y_val = y_val
    for epoch in pbar:
        total_loss = 0
        order = np.random.permutation(len(X_train_rescaled))
        for i in range(0, len(X_train_rescaled), batch_size):
            batch_x = X_train_rescaled[order[i : i + batch_size]]
            batch_y = y_train[order[i : i + batch_size]]

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()

        with torch.no_grad():
            val_pred = model.predict_proba(val_x).cpu().numpy()
            val_auc = roc_auc_score(y_val.cpu(), val_pred)
        model.train()

        pbar.set_postfix({"Loss": f"{total_loss:.4f}", "Val AUC": f"{val_auc:.4f}"})

        # Check for improvement
        if val_auc > best_val_auc + min_delta:
            best_val_auc = val_auc
            best_model = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve == patience:
            model.load_state_dict(best_model)
            return model, best_val_auc

    if best_model is not None:
        model.load_state_dict(best_model)
        return model, best_val_auc
    return model, 0.5


from itertools import product


def train_heldout(
    network,
    node_features=None,
    param_ranges=None,
    max_patience=100,
    negative_edge_sampler="uniform",
    with_degree=False,
    device="cpu",
    lr=0.001,
    epochs=500,
    batch_size=1024 * 5,
):
    """
    Train MLP model with hyperparameter tuning using held-out validation set.

    Args:
        network (scipy.sparse.csr_matrix): Sparse adjacency matrix
        node_features (numpy.ndarray, optional): Node features matrix
        param_ranges (dict, optional): Dict of parameter ranges to try
        max_patience (int): Early stopping patience
        negative_edge_sampler (str): Method for sampling negative edges
        with_degree (bool): Whether to include degree features
        device (str): Computing device ('cpu' or 'cuda')

    Returns:
        best_model (MLP): Trained model with best parameters
        best_params (dict): Best parameter configuration
        best_val_score (float): Best validation score achieved
    """
    # Default parameter ranges if none provided
    if param_ranges is None:
        param_ranges = {
            "hidden_layers": [[32, 32], [64, 64]],
            "dropout_rate": [0.2, 0.5],
            "activation": ["leaky_relu"],
        }

    # Generate all parameter combinations
    param_combinations = list(product(*param_ranges.values()))

    # Split positive edges into train/val sets
    pos_src, pos_trg, _ = sparse.find(sparse.triu(network, 1))
    train_src, val_src, train_trg, val_trg = train_test_split(
        pos_src, pos_trg, test_size=0.25
    )

    # Sample negative edges for training and validation
    neg_sampler = NegativeEdgeSampler(negative_edge_sampler=negative_edge_sampler)
    neg_sampler.fit(network)

    neg_src, neg_trg = neg_sampler.sampling(source_nodes=train_src, size=len(train_src))
    val_neg_src, val_neg_trg = neg_sampler.sampling(
        source_nodes=val_src, size=len(val_src)
    )

    # Prepare training and validation data
    X_train = compute_network_stats(
        network,
        np.concatenate([train_src, neg_src]),
        np.concatenate([train_trg, neg_trg]),
        with_degree=with_degree,
    )
    y_train = np.concatenate([np.ones(len(train_src)), np.zeros(len(neg_src))])

    X_val = compute_network_stats(
        network,
        np.concatenate([val_src, val_neg_src]),
        np.concatenate([val_trg, val_neg_trg]),
        with_degree=with_degree,
    )
    y_val = np.concatenate([np.ones(len(val_src)), np.zeros(len(val_neg_src))])

    # Shuffle the training and validation data
    train_indices = np.random.permutation(len(y_train))
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]

    val_indices = np.random.permutation(len(y_val))
    X_val = X_val[val_indices]
    y_val = y_val[val_indices]

    # Store results
    results = []
    best_val_score = 0

    # Convert data to tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)

    # Try each parameter combination
    for params in param_combinations:
        model = MLP(
            with_degree=with_degree,
            hidden_layers=params[0],
            dropout_rate=params[1],
        ).to(device)

        # Train model
        trained_model, val_score = train_mlp_with_early_stopping(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=epochs,
            patience=max_patience,
            device=device,
            lr=lr,
            batch_size=batch_size,
        )

        if val_score > best_val_score:
            best_val_score = val_score
            best_params = dict(zip(param_ranges.keys(), params))

        results.append(
            {"params": dict(zip(param_ranges.keys(), params)), "val_score": val_score}
        )

        # Clear memory
        del trained_model
        torch.cuda.empty_cache()

    # Train final model with best params
    final_model = MLP(
        with_degree=with_degree,
        hidden_layers=best_params["hidden_layers"],
        dropout_rate=best_params["dropout_rate"],
    ).to(device)

    best_model, _ = train_mlp_with_early_stopping(
        final_model,
        X_train,
        y_train,
        X_val,
        y_val,
        patience=max_patience,
        device=device,
    )

    return best_model, best_params


# with_degree = False
#
# import networkx as nx
#
# G = nx.karate_club_graph()
# A = nx.adjacency_matrix(G)
# labels = np.unique([d[1]["club"] for d in G.nodes(data=True)], return_inverse=True)[1]
#
## Test script for MLP training
# A = sparse.csr_matrix(nx.adjacency_matrix(G))
#
## Create train/test split
# from gnn_tools.LinkPredictionDataset import TrainTestEdgeSplitter
#
# splitter = TrainTestEdgeSplitter(fraction=0.2)
# splitter.fit(A)
# train_src, train_trg = splitter.train_edges_
# test_src, test_trg = splitter.test_edges_
#
# train_net = sparse.csr_matrix(
#    (np.ones(len(train_src)), (train_src, train_trg)), shape=A.shape
# )
# test_net = sparse.csr_matrix(
#    (np.ones(len(test_src)), (test_src, test_trg)), shape=A.shape
# )
#
## Generate negative test edges
# sampler = NegativeEdgeSampler(negative_edge_sampler="uniform")
# sampler.fit(A)
# neg_src, neg_trg = sampler.sampling(source_nodes=test_src, size=len(test_src))
#
## Combine positive and negative edges
# test_src = np.concatenate([test_src, neg_src])
# test_trg = np.concatenate([test_trg, neg_trg])
# test_labels = np.concatenate(
#    [np.ones(len(splitter.test_edges_[0])), np.zeros(len(neg_src))]
# )
#
## Train MLP model
# device = "cuda:1"
#
# X_test = compute_network_stats(train_net, test_src, test_trg, with_degree=with_degree)
# X_test = torch.FloatTensor(X_test).to(device)
# y_test = torch.FloatTensor(test_labels).to(device)
#
## Train with default parameters
# trained_model, val_score = train_heldout(
#    network=train_net,
#    with_degree=with_degree,
#    max_patience=100,
#    device="cuda:1",
#    # epochs=1000,
# )
# trained_model.eval()
## %%
## Make predictions
# src, trg = np.triu_indices(A.shape[0], k=1)
## X_test_rescaled = trained_model.get_rescaled_features(X_test)
## preds = trained_model.forward(X_test_rescaled).cpu().detach().numpy()
# preds = trained_model.forward_edges(train_net, src, trg).cpu().detach().numpy()
## auc_score = roc_auc_score(y_test.cpu(), preds)
## print(f"Test AUC: {auc_score:.4f}")
# B = np.zeros((A.shape[0], A.shape[0]))
# B[src, trg] = preds
# B[trg, src] = preds
# import seaborn as sns
#
# sns.heatmap(B)
## %%
# X_test_rescaled = trained_model.get_rescaled_features(X_test)
# preds = trained_model.forward(X_test_rescaled).cpu().detach().numpy()
##preds = trained_model.forward_edges(train_net, src, trg).cpu().detach().numpy()
# auc_score = roc_auc_score(y_test.cpu(), preds)
# print(f"Test AUC: {auc_score:.4f}")

# %%
