"""
Modified runner for BUDDY GNN with custom dataset support - uses provided test edges
"""

import copy
import warnings
import sys
from math import inf
import numpy as np
import torch
from scipy.sparse import SparseEfficiencyWarning
from torch_geometric.data import Data
import wandb
from buddy.models.elph import BUDDY
from buddy.utils import select_embedding
from buddy.runners.train import train_buddy
from buddy.runners.inference import test
from buddy.data import get_loaders, get_train_loaders
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, List
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from buddy.datasets.elph import HashDataset
import os
import json
from dataclasses import asdict
from typing import Tuple, Optional
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data
import torch


# Suppress SparseEfficiencyWarning
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

# Set torch print options
torch.set_printoptions(precision=4)

# Add parent directory to system path
sys.path.insert(0, "..")


@dataclass
class BuddyConfig:
    # Dataset settings
    val_pct: float = 0.1
    test_pct: float = 0.0
    train_samples: float = inf
    val_samples: float = inf
    test_samples: float = inf
    preprocessing: Optional[str] = None
    sign_k: int = 0
    load_features: bool = False
    load_hashes: bool = False
    cache_subgraph_features: bool = False
    train_cache_size: float = inf
    year: int = 0

    # GNN settings
    model: str = "BUDDY"
    dataset_name: str = "custom"
    hidden_channels: int = 1024
    batch_size: int = 1024
    eval_batch_size: int = 1000000
    label_dropout: float = 0.5
    feature_dropout: float = 0.5
    sign_dropout: float = 0.5
    save_model: bool = False
    feature_prop: str = "gcn"

    # SEAL settings
    dropout: float = 0.5
    num_seal_layers: int = 3
    sortpool_k: float = 0.6
    label_pooling: str = "add"
    seal_pooling: str = "edge"

    # Subgraph settings
    num_hops: int = 1
    ratio_per_hop: float = 1.0
    max_nodes_per_hop: Optional[int] = None
    node_label: str = "drnl"
    max_dist: int = 4
    max_z: int = 1000
    use_feature: bool = False
    use_struct_feature: bool = True
    use_edge_weight: bool = False

    # Training settings
    lr: float = 0.001  # 0.0001
    weight_decay: float = 0
    epochs: int = 100
    num_workers: int = 4
    num_negs: int = 1
    train_node_embedding: bool = True
    propagate_embeddings: bool = False
    loss: str = "bce"
    add_normed_features: Optional[bool] = True
    use_RA: bool = False

    # SEAL specific settings
    dynamic_train: bool = False
    dynamic_val: bool = False
    dynamic_test: bool = False
    pretrained_node_embedding: Optional[str] = None

    # Testing settings
    reps: int = 1
    use_valedges_as_input: bool = False
    eval_steps: int = 1
    log_steps: int = 1
    eval_metric: str = "hits"
    K: int = 50  # 100

    # Hash settings
    use_zero_one: bool = False
    floor_sf: bool = False
    hll_p: int = 8
    minhash_num_perm: int = 64  # 128
    max_hash_hops: int = 2
    subgraph_feature_batch_size: int = 11000000

    # Wandb settings
    wandb: bool = False
    use_wandb_offline: bool = False
    wandb_sweep: bool = False
    wandb_watch_grad: bool = False
    wandb_track_grad_flow: bool = False
    wandb_entity: str = "link-prediction"
    wandb_project: str = "link-prediction"
    wandb_group: str = "testing"
    wandb_run_name: Optional[str] = None
    wandb_output_dir: str = "./wandb_output"
    wandb_log_freq: int = 1
    wandb_epoch_list: List[int] = None
    log_features: bool = False

    num_features: int = 0
    num_nodes: int = 0

    def __post_init__(self):
        if self.wandb_epoch_list is None:
            self.wandb_epoch_list = [0, 1, 2, 4, 8, 16]

        # Validation checks
        if self.max_hash_hops == 1 and not self.use_zero_one:
            print(
                "WARNING: (0,1) feature knock out is not supported for 1 hop. Running with all features"
            )

        if self.dataset_name == "ogbl-ddi":
            self.use_feature = False
            assert (
                self.sign_k > 0
            ), "--sign_k must be set to > 0 i.e. 1,2 or 3 for ogbl-ddi"

    @classmethod
    def from_args(cls, args):
        return cls(**vars(args))

    def copy(self):
        return copy.deepcopy(self)


from itertools import product


def train_heldout(
    adj_matrix,
    node_features=None,
    config: BuddyConfig = None,
    model_file_path=None,
    max_patience=5,
    param_ranges=None,
    device="cpu",
):
    # Use provided param_ranges or default to an empty dict
    if param_ranges is None:
        param_ranges = {
            "num_hops": [2],
            "hidden_channels": [256, 1024],
            "feature_dropout": [0.05, 0.2],
            # "use_RA": [True, False],
        }

    # Generate all parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    param_combinations = list(product(*param_values))

    # Store results for all combinations
    results = []

    # Try each parameter combination
    base_config = config.copy()
    best_val_score = 0
    for combination in param_combinations:
        args = base_config.copy()
        for name, value in zip(param_names, combination):
            setattr(args, name, value)
        dataset, args, train_loader, train_eval_loader, val_loader = compile_data(
            adj_matrix=adj_matrix,
            device=device,
            node_features=node_features,
            config=args,
        )

        model, metrics = train_with_early_stopping(
            dataset=dataset,
            args=args,
            device=device,
            train_loader=train_loader,
            train_eval_loader=train_eval_loader,
            val_loader=val_loader,
            model_file_path=None,
            max_patience=max_patience,
        )

        score = list(metrics.values())[0]

        if score > best_val_score:
            best_val_score = score

            # Store results
            results.append(
                {
                    "args": args.copy(),
                    "val_score": best_val_score,
                }
            )
        # Clear model and metrics to free memory after each iteration
        del model
        torch.cuda.empty_cache()

    # Sort results by validation score
    results.sort(key=lambda x: x["val_score"], reverse=True)

    best_args = results[0]["args"]
    best_args.val_pct = 0.0

    best_model = train(
        adj_matrix,
        node_features=node_features,
        config=best_args,
        model_file_path=model_file_path,
        device=device,
    )

    return best_model, best_args


def train(
    adj_matrix,
    node_features=None,
    config: BuddyConfig = None,
    model_file_path=None,
    device="cpu",
):
    """
    Wrapper function to train and evaluate BUDDY model using original infrastructure
    """
    # Create PyG Data object
    dataset, args, train_loader, train_eval_loader = compile_data(
        adj_matrix=adj_matrix,
        device=device,
        node_features=node_features,
        config=config,
        only_train_loader=True
    )

    # Initialize model
    emb = select_embedding(args, dataset.data.num_nodes, device)
    model = BUDDY(args, dataset.num_features, node_embedding=emb).to(device)
    args = model.get_config()

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop using original infrastructure
    pbar = tqdm(range(args.epochs))
    for _ in pbar:
        # Train
        loss = train_buddy(model, optimizer, train_loader, args, device)

        pbar.set_postfix(
            loss=f"{loss:.4f}",
        )

    if args.wandb:
        wandb.finish()
    if model_file_path is not None:
        save_model(model, args, model_file_path)
    return model


def compile_data(adj_matrix, device, node_features=None, config: BuddyConfig = None, only_train_loader = False):
    """
    Compile data into format expected by get_train_loaders

    Parameters
    ----------
    adj_matrix : scipy.sparse.csr_matrix
        Adjacency matrix of the graph
    node_features : numpy.ndarray, optional
        Node features, by default None
    config : BuddyConfig, optional
        Configuration object, by default None
    """

    args = BuddyConfig() if config is None else config
    # args = default_args
    device = torch.device(device)

    if node_features is None:
        node_features = np.zeros((adj_matrix.shape[0], 0))

    # Create PyG Data object
    edge_index = torch.from_numpy(np.array(adj_matrix.nonzero())).long()
    x = torch.from_numpy(node_features).float()
    data = Data(x=x, edge_index=edge_index)

    # Split data using RandomLinkSplit
    print(args.test_pct)
    transform = RandomLinkSplit(
        num_val=0 if only_train_loader else args.val_pct,
        num_test=0 if only_train_loader else args.test_pct,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
    )

    train_data, val_data, test_data = transform(data)

    # Create dataset object for BUDDY
    class CustomDataset:
        def __init__(self, root, x, data, edge_index, num_features):
            self.root = root
            self.x = x
            self.data = data
            self.edge_index = edge_index
            self.num_features = num_features
            self.data = data  # Store the original data

    dataset = CustomDataset(
        root="./",
        data=data,
        x=x,
        edge_index=train_data.edge_index,
        num_features=node_features.shape[1],
    )

    # Create splits in format expected by get_hashed_train_val_test_datasets
    splits = {
        "train": train_data,  # Keep the full PyG Data objects
        "valid": val_data,
        # "test": test_data,
    }

    # Get data loaders using original infrastructure
    if only_train_loader:
        train_loader, train_eval_loader = get_train_loaders(
            args, dataset, splits, directed=False
        )
        return dataset, args, train_loader, train_eval_loader
    else:
        train_loader, train_eval_loader, val_loader = get_loaders(
            args, dataset, splits, directed=False
        )
        return dataset, args, train_loader, train_eval_loader, val_loader


def train_with_early_stopping(
    # adj_matrix,
    # node_features=None,
    # config: BuddyConfig = None,
    dataset,
    args,
    device,
    train_loader,
    train_eval_loader,
    val_loader,
    model_file_path=None,
    max_patience=30,
):
    """
    Wrapper function to train and evaluate BUDDY model using original infrastructure
    """

    # Initialize model
    emb = select_embedding(args, dataset.data.num_nodes, device)
    model = BUDDY(args, dataset.num_features, node_embedding=emb).to(device)
    args = model.get_config()

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop using original infrastructure
    best_val_metric = 0
    patience = 0
    pbar = tqdm(range(args.epochs))
    for _ in pbar:
        # Train
        loss = train_buddy(model, optimizer, train_loader, args, device)

        # Evaluate
        results = test(
            model,
            None,
            train_eval_loader,
            val_loader,
            # test_loader,
            args,
            device,
            eval_metric="auc",
        )

        # Get validation metric
        val_metric = list(results.values())[0]

        # Update best metrics
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            patience = 0
        else:
            patience += 1

        pbar.set_postfix(
            loss=f"{loss:.4f}",
            valid=f"{100 * val_metric:.2f}%",
            best_valid=f"{100 * best_val_metric:.2f}%",
        )

        if patience >= max_patience:
            print("Early stopping triggered")
            break

    if args.wandb:
        wandb.finish()
    if model_file_path is not None:
        save_model(model, args, model_file_path)
    return model, {"val_metric": best_val_metric}


def predict_edge_likelihood(
    model, adj_matrix, candidate_edges, args, device="cpu", node_features=None
):
    """
    Predicts likelihood scores for candidate edges in a network.

    Args:
        model: Trained BUDDY/ELPH model
        network_data: PyG Data object containing the network structure (x, edge_index)
        candidate_edges: [2, num_edges] tensor of edges to evaluate
        args: Configuration object
        device: Computing device

    Returns:
        torch.Tensor: Prediction scores for candidate edges
    """
    model.eval()
    model.to(device)
    edge_index = torch.from_numpy(np.array(adj_matrix.nonzero())).long()
    if node_features is not None:
        x = torch.from_numpy(node_features).float()
    else:
        x = torch.zeros((adj_matrix.shape[0], 0)).float()

    network_data = Data(x=x, edge_index=edge_index)

    # Create dataset for candidate edges
    dataset = HashDataset(
        root="elph_data",
        split="valid",
        data=network_data,
        pos_edges=candidate_edges.long().t(),
        neg_edges=torch.empty((0, 2)).long(),
        args=args,
        use_coalesce=False,
        directed=False,
    )

    # Setup batched prediction
    loader = DataLoader(
        range(candidate_edges.size(1)), batch_size=args.eval_batch_size, shuffle=False
    )

    # Get node embeddings if model uses them
    emb = None
    if model.node_embedding is not None:
        emb = (
            model.propagate_embeddings_func(network_data.edge_index)
            if args.propagate_embeddings
            else model.node_embedding.weight
        )

    # Make predictions
    predictions = []
    with torch.no_grad():
        for indices in loader:
            # Get current batch edges
            curr_edges = candidate_edges.t()[indices]

            # Prepare inputs
            inputs = {
                "subgraph_features": (
                    dataset.subgraph_features[indices]
                    if args.use_struct_feature
                    else torch.zeros_like(dataset.subgraph_features[indices])
                ),
                "node_features": dataset.x[curr_edges],
                "src_degrees": dataset.degrees[curr_edges][:, 0],
                "dst_degrees": dataset.degrees[curr_edges][:, 1],
                "RA": dataset.RA[indices] if args.use_RA else None,
                "embeddings": None if emb is None else emb[curr_edges],
            }

            # Move everything to device
            inputs = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in inputs.items()
            }

            # Get predictions
            logits = model(
                inputs["subgraph_features"],
                inputs["node_features"],
                inputs["src_degrees"],
                inputs["dst_degrees"],
                inputs["RA"],
                inputs["embeddings"],
            )
            predictions.append(logits.view(-1).cpu())

    return torch.cat(predictions)


def load_model(
    model_path: str, data=None, device: str = "cpu"
) -> Tuple[torch.nn.Module, BuddyConfig]:
    """
    Loads a saved BUDDY/ELPH model along with its configuration.

    Args:
        model_path: Path to the directory containing model files
        data: Optional PyG Data object containing the network structure
        device: Computing device ("cpu" or "cuda")

    Returns:
        tuple: (loaded_model, buddy_config)

    Directory structure:
        model_path/
            └── model.pt          # Model state dict
            └── config.json       # BuddyConfig configuration
    """
    # Check paths
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    model_file = os.path.join(model_path, "model.pt")
    config_file = os.path.join(model_path, "config.json")

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model weights not found: {model_file}")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Model config not found: {config_file}")

    # Load configuration
    with open(config_file, "r") as f:
        config_dict = json.load(f)
        config = BuddyConfig(**config_dict)

    # Initialize model based on type
    emb = select_embedding(config, config.num_nodes, device)
    model = BUDDY(
        config,
        num_features=config.num_features,
        node_embedding=emb,
    ).to(device)

    # Load state dict
    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    return model, config


def save_model(model: torch.nn.Module, config: BuddyConfig, save_path: str):
    """
    Saves model and its configuration.

    Args:
        model: BUDDY/ELPH model to save
        config: BuddyConfig object
        save_path: Directory path to save model files
    """
    os.makedirs(save_path, exist_ok=True)

    # Save model weights
    model_file = os.path.join(save_path, "model.pt")
    torch.save(model.state_dict(), model_file)

    # Save configuration
    config_file = os.path.join(save_path, "config.json")
    config_dict = asdict(config)
    with open(config_file, "w") as f:
        json.dump(config_dict, f, indent=4)
