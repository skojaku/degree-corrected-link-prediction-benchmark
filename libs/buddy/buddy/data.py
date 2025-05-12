"""
Read and split ogb and planetoid datasets
"""

import os
import time
from typing import Optional, Tuple, Union
import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import Tensor
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import add_self_loops, negative_sampling, to_undirected

# from torch_geometric.utils.negative_sampling import *
# from torch_geometric.utils.negative_sampling import vector_to_edge_index, edge_index_to_vector, sample
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.loader import DataLoader as pygDataLoader
import wandb

from buddy.utils import ROOT_DIR, get_same_source_negs, neighbors
from buddy.lcc import get_largest_connected_component, remap_edges, get_node_mapper
from buddy.datasets.seal import get_train_val_test_datasets
from buddy.datasets.elph import (
    get_hashed_train_val_test_datasets,
    make_train_eval_data,
    get_hashed_train_val_datasets,
    get_hashed_train_datasets,
)


def get_train_loaders(args, dataset, splits, directed):
    train_data = splits["train"]
    train_dataset = get_hashed_train_datasets(dataset, train_data, args, directed)

    dl = DataLoader if args.model in {"ELPH", "BUDDY"} else pygDataLoader
    train_loader = dl(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    # as the val and test edges are often sampled they also need to be shuffled
    # the citation2 dataset has specific negatives for each positive and so can't be shuffled
    if (args.dataset_name == "ogbl-citation2") and (args.model in {"ELPH", "BUDDY"}):
        train_eval_loader = dl(
            make_train_eval_data(
                args, train_dataset, train_data.num_nodes, n_pos_samples=5000
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
    else:
        # todo change this so that eval doesn't have to use the full training set
        train_eval_loader = train_loader

    return train_loader, train_eval_loader


def get_loaders(args, dataset, splits, directed):
    train_data, val_data = splits["train"], splits["valid"]
    train_dataset, val_dataset = get_hashed_train_val_datasets(
        dataset, train_data, val_data, args, directed
    )

    dl = DataLoader if args.model in {"ELPH", "BUDDY"} else pygDataLoader
    train_loader = dl(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    # as the val and test edges are often sampled they also need to be shuffled
    # the citation2 dataset has specific negatives for each positive and so can't be shuffled
    shuffle_val = False if args.dataset_name.startswith("ogbl-citation") else True
    val_loader = dl(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle_val,
        num_workers=args.num_workers,
    )
    shuffle_test = False if args.dataset_name.startswith("ogbl-citation") else True
    # test_loader = dl(test_dataset, batch_size=args.batch_size, shuffle=shuffle_test,
    #                 num_workers=args.num_workers)
    if (args.dataset_name == "ogbl-citation2") and (args.model in {"ELPH", "BUDDY"}):
        train_eval_loader = dl(
            make_train_eval_data(
                args, train_dataset, train_data.num_nodes, n_pos_samples=5000
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
    else:
        # todo change this so that eval doesn't have to use the full training set
        train_eval_loader = train_loader

    return train_loader, train_eval_loader, val_loader


def get_data(args):
    """
    Read the dataset and generate train, val and test splits.
    For GNN link prediction edges play 2 roles 1/ message passing edges 2/ supervision edges
    - train message passing edges = train supervision edges
    - val message passing edges = train supervision edges
    val supervision edges are disjoint from the training edges
    - test message passing edges = val supervision + train message passing (= val message passing)
    test supervision edges are disjoint from both val and train supervision edges
    :param args: arguments Namespace object
    :return: dataset, dic splits, bool directed, str eval_metric
    """
    include_negatives = True
    dataset_name = args.dataset_name
    val_pct = args.val_pct
    test_pct = args.test_pct
    use_lcc_flag = True
    directed = False
    eval_metric = "hits"
    path = os.path.join(ROOT_DIR, "dataset", dataset_name)
    print(f"reading data from: {path}")
    if dataset_name.startswith("ogbl"):
        use_lcc_flag = False
        dataset = PygLinkPropPredDataset(name=dataset_name, root=path)
        if dataset_name == "ogbl-ddi":
            dataset.data.x = torch.ones((dataset.data.num_nodes, 1))
            dataset.data.edge_weight = torch.ones(
                dataset.data.edge_index.size(1), dtype=int
            )
    else:
        dataset = Planetoid(path, dataset_name)

    # set the metric
    if dataset_name.startswith("ogbl-citation"):
        eval_metric = "mrr"
        directed = True

    if use_lcc_flag:
        dataset = use_lcc(dataset)

    undirected = not directed

    if dataset_name.startswith("ogbl"):  # use the built in splits
        data = dataset[0]
        split_edge = dataset.get_edge_split()
        if (
            dataset_name == "ogbl-collab" and args.year > 0
        ):  # filter out training edges before args.year
            data, split_edge = filter_by_year(data, split_edge, args.year)
        splits = get_ogb_data(data, split_edge, dataset_name, args.num_negs)
    else:  # make random splits
        transform = RandomLinkSplit(
            is_undirected=undirected,
            num_val=val_pct,
            num_test=test_pct,
            add_negative_train_samples=include_negatives,
        )
        train_data, val_data, test_data = transform(dataset.data)
        splits = {"train": train_data, "valid": val_data, "test": test_data}

    return dataset, splits, directed, eval_metric


def filter_by_year(data, split_edge, year):
    """
    remove edges before year from data and split edge
    @param data: pyg Data, pyg SplitEdge
    @param split_edges:
    @param year: int first year to use
    @return: pyg Data, pyg SplitEdge
    """
    selected_year_index = torch.reshape(
        (split_edge["train"]["year"] >= year).nonzero(as_tuple=False), (-1,)
    )
    split_edge["train"]["edge"] = split_edge["train"]["edge"][selected_year_index]
    split_edge["train"]["weight"] = split_edge["train"]["weight"][selected_year_index]
    split_edge["train"]["year"] = split_edge["train"]["year"][selected_year_index]
    train_edge_index = split_edge["train"]["edge"].t()
    # create adjacency matrix
    new_edges = to_undirected(
        train_edge_index, split_edge["train"]["weight"], reduce="add"
    )
    new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
    data.edge_index = new_edge_index
    data.edge_weight = new_edge_weight.unsqueeze(-1)
    return data, split_edge


def get_ogb_data(data, split_edge, dataset_name, num_negs=1):
    """
    ogb datasets come with fixed train-val-test splits and a fixed set of negatives against which to evaluate the test set
    The dataset.data object contains all of the nodes, but only the training edges
    @param dataset:
    @param use_valedges_as_input:
    @return:
    """
    if num_negs == 1:
        negs_name = f"{ROOT_DIR}/dataset/{dataset_name}/negative_samples.pt"
    else:
        negs_name = f"{ROOT_DIR}/dataset/{dataset_name}/negative_samples_{num_negs}.pt"
    print(f"looking for negative edges at {negs_name}")
    if os.path.exists(negs_name):
        print("loading negatives from disk")
        train_negs = torch.load(negs_name)
    else:
        print("negatives not found on disk. Generating negatives")
        train_negs = get_ogb_train_negs(
            split_edge, data.edge_index, data.num_nodes, num_negs, dataset_name
        )
        torch.save(train_negs, negs_name)
    # else:
    #     train_negs = get_ogb_train_negs(split_edge, data.edge_index, data.num_nodes, num_negs, dataset_name)
    splits = {}
    for key in split_edge.keys():
        # the ogb datasets come with test and valid negatives, but you have to cook your own train negs
        neg_edges = train_negs if key == "train" else None
        edge_label, edge_label_index = make_obg_supervision_edges(
            split_edge, key, neg_edges
        )
        # use the validation edges for message passing at test time
        # according to the rules https://ogb.stanford.edu/docs/leader_rules/ only collab can use val edges at test time
        if key == "test" and dataset_name == "ogbl-collab":
            vei, vw = to_undirected(
                split_edge["valid"]["edge"].t(), split_edge["valid"]["weight"]
            )
            edge_index = torch.cat([data.edge_index, vei], dim=1)
            edge_weight = torch.cat([data.edge_weight, vw.unsqueeze(-1)], dim=0)
        else:
            edge_index = data.edge_index
            if hasattr(data, "edge_weight"):
                edge_weight = data.edge_weight
            else:
                edge_weight = torch.ones(data.edge_index.shape[1])
        splits[key] = Data(
            x=data.x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_label=edge_label,
            edge_label_index=edge_label_index,
        )
    return splits


def get_ogb_pos_edges(split_edge, split):
    if "edge" in split_edge[split]:
        pos_edge = split_edge[split]["edge"]
    elif "source_node" in split_edge[split]:
        pos_edge = torch.stack(
            [split_edge[split]["source_node"], split_edge[split]["target_node"]], dim=1
        )
    else:
        raise NotImplementedError
    return pos_edge


def get_ogb_train_negs(
    split_edge, edge_index, num_nodes, num_negs=1, dataset_name=None
):
    """
    for some inexplicable reason ogb datasets split_edge object stores edge indices as (n_edges, 2) tensors
    @param split_edge:

    @param edge_index: A [2, num_edges] tensor
    @param num_nodes:
    @param num_negs: the number of negatives to sample for each positive
    @return: A [num_edges * num_negs, 2] tensor of negative edges
    """
    pos_edge = get_ogb_pos_edges(split_edge, "train").t()
    if dataset_name is not None and dataset_name.startswith("ogbl-citation"):
        neg_edge = get_same_source_negs(num_nodes, num_negs, pos_edge)
    else:  # any source is fine
        new_edge_index, _ = add_self_loops(edge_index)
        neg_edge = negative_sampling(
            new_edge_index,
            num_nodes=num_nodes,
            num_neg_samples=pos_edge.size(1) * num_negs,
        )
    return neg_edge.t()


def make_obg_supervision_edges(split_edge, split, neg_edges=None):
    if neg_edges is not None:
        neg_edges = neg_edges
    else:
        if "edge_neg" in split_edge[split]:
            neg_edges = split_edge[split]["edge_neg"]
        elif "target_node_neg" in split_edge[split]:
            n_neg_nodes = split_edge[split]["target_node_neg"].shape[1]
            neg_edges = torch.stack(
                [
                    split_edge[split]["source_node"]
                    .unsqueeze(1)
                    .repeat(1, n_neg_nodes)
                    .ravel(),
                    split_edge[split]["target_node_neg"].ravel(),
                ]
            ).t()
        else:
            raise NotImplementedError

    pos_edges = get_ogb_pos_edges(split_edge, split)
    n_pos, n_neg = pos_edges.shape[0], neg_edges.shape[0]
    edge_label = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)], dim=0)
    edge_label_index = torch.cat([pos_edges, neg_edges], dim=0).t()
    return edge_label, edge_label_index


def use_lcc(dataset):
    lcc = get_largest_connected_component(dataset)

    x_new = dataset.data.x[lcc]
    y_new = dataset.data.y[lcc]

    row, col = dataset.data.edge_index.numpy()
    edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
    edges = remap_edges(edges, get_node_mapper(lcc))

    data = Data(
        x=x_new,
        edge_index=torch.LongTensor(edges),
        y=y_new,
        train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
    )
    dataset.data = data
    return dataset


def sample_hard_negatives(
    edge_index: Tensor,
    num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
    num_neg_samples: Optional[int] = None,
) -> Tensor:
    """
    Sample hard negatives for each edge in edge_index
    @param edge_index:
    @return:
    """
    if num_nodes is None:
        num_nodes = maybe_num_nodes(edge_index)
    # get the size of the population of edges and the index of the existing edges into this population
    idx, population = edge_index_to_vector(
        edge_index, (num_nodes, num_nodes), bipartite=False
    )
    # for each node, get all of the neighbours and produce all edges that have that node as a common neigbour
    common_neighbour_edges = []
    for node in range(num_nodes):
        neighbours = edge_index[1, edge_index[0] == node]
        # get all edges that have a common neighbour with node
        edges = list(itertools.combinations(neighbours, 2))
        common_neighbour_edges.extend(edges)
    unique_common_neighbour_edges = list(set(common_neighbour_edges))
    # get the unique edges that are not in the graph
    # 1. turn this into an edge index
    # 2. get the index of the common neighbour edges into the population
    # 3. get common neighbours that are not in the graph
    # 4. maybe sample

    # get the index of the common neighbour edges into the population

    # sample num_neg_samples edges from the population of common neighbour edges
    idx = idx.to("cpu")
    for _ in range(3):  # Number of tries to sample negative indices.
        rnd = sample(population, num_neg_samples, device="cpu")
        mask = np.isin(rnd, idx)
        if neg_idx is not None:
            mask |= np.isin(rnd, neg_idx.to("cpu"))
        mask = torch.from_numpy(mask).to(torch.bool)
        rnd = rnd[~mask].to(edge_index.device)
        neg_idx = rnd if neg_idx is None else torch.cat([neg_idx, rnd])
        if neg_idx.numel() >= num_neg_samples:
            neg_idx = neg_idx[:num_neg_samples]
            break


import random
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from torch_geometric.utils import coalesce, cumsum, degree, remove_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes


def negative_sampling(
    edge_index: Tensor,
    num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
    num_neg_samples: Optional[int] = None,
    method: str = "sparse",
    force_undirected: bool = False,
) -> Tensor:
    r"""Samples random negative edges of a graph given by :attr:`edge_index`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int or Tuple[int, int], optional): The number of nodes,
            *i.e.* :obj:`max_val + 1` of :attr:`edge_index`.
            If given as a tuple, then :obj:`edge_index` is interpreted as a
            bipartite graph with shape :obj:`(num_src_nodes, num_dst_nodes)`.
            (default: :obj:`None`)
        num_neg_samples (int, optional): The (approximate) number of negative
            samples to return.
            If set to :obj:`None`, will try to return a negative edge for every
            positive edge. (default: :obj:`None`)
        method (str, optional): The method to use for negative sampling,
            *i.e.* :obj:`"sparse"` or :obj:`"dense"`.
            This is a memory/runtime trade-off.
            :obj:`"sparse"` will work on any graph of any size, while
            :obj:`"dense"` can perform faster true-negative checks.
            (default: :obj:`"sparse"`)
        force_undirected (bool, optional): If set to :obj:`True`, sampled
            negative edges will be undirected. (default: :obj:`False`)

    :rtype: LongTensor

    Examples:

        >>> # Standard usage
        >>> edge_index = torch.as_tensor([[0, 0, 1, 2],
        ...                               [0, 1, 2, 3]])
        >>> negative_sampling(edge_index)
        tensor([[3, 0, 0, 3],
                [2, 3, 2, 1]])

        >>> # For bipartite graph
        >>> negative_sampling(edge_index, num_nodes=(3, 4))
        tensor([[0, 2, 2, 1],
                [2, 2, 1, 3]])
    """
    assert method in ["sparse", "dense"]

    size = num_nodes
    bipartite = isinstance(size, (tuple, list))
    size = maybe_num_nodes(edge_index) if size is None else size
    size = (size, size) if not bipartite else size
    force_undirected = False if bipartite else force_undirected

    idx, population = edge_index_to_vector(
        edge_index, size, bipartite, force_undirected
    )

    if idx.numel() >= population:
        return edge_index.new_empty((2, 0))

    if num_neg_samples is None:
        num_neg_samples = edge_index.size(1)
    if force_undirected:
        num_neg_samples = num_neg_samples // 2

    prob = 1.0 - idx.numel() / population  # Probability to sample a negative.
    sample_size = int(1.1 * num_neg_samples / prob)  # (Over)-sample size.

    neg_idx = None
    if method == "dense":
        # The dense version creates a mask of shape `population` to check for
        # invalid samples.
        mask = idx.new_ones(population, dtype=torch.bool)
        mask[idx] = False
        for _ in range(3):  # Number of tries to sample negative indices.
            rnd = sample(population, sample_size, idx.device)
            rnd = rnd[mask[rnd]]  # Filter true negatives.
            neg_idx = rnd if neg_idx is None else torch.cat([neg_idx, rnd])
            if neg_idx.numel() >= num_neg_samples:
                neg_idx = neg_idx[:num_neg_samples]
                break
            mask[neg_idx] = False

    else:  # 'sparse'
        # The sparse version checks for invalid samples via `np.isin`.
        idx = idx.to("cpu")
        for _ in range(3):  # Number of tries to sample negative indices.
            rnd = sample(population, sample_size, device="cpu")
            mask = np.isin(rnd, idx)
            if neg_idx is not None:
                mask |= np.isin(rnd, neg_idx.to("cpu"))
            mask = torch.from_numpy(mask).to(torch.bool)
            rnd = rnd[~mask].to(edge_index.device)
            neg_idx = rnd if neg_idx is None else torch.cat([neg_idx, rnd])
            if neg_idx.numel() >= num_neg_samples:
                neg_idx = neg_idx[:num_neg_samples]
                break

    return vector_to_edge_index(neg_idx, size, bipartite, force_undirected)


def batched_negative_sampling(
    edge_index: Tensor,
    batch: Union[Tensor, Tuple[Tensor, Tensor]],
    num_neg_samples: Optional[int] = None,
    method: str = "sparse",
    force_undirected: bool = False,
) -> Tensor:
    r"""Samples random negative edges of multiple graphs given by
    :attr:`edge_index` and :attr:`batch`.

    Args:
        edge_index (LongTensor): The edge indices.
        batch (LongTensor or Tuple[LongTensor, LongTensor]): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example.
            If given as a tuple, then :obj:`edge_index` is interpreted as a
            bipartite graph connecting two different node types.
        num_neg_samples (int, optional): The number of negative samples to
            return. If set to :obj:`None`, will try to return a negative edge
            for every positive edge. (default: :obj:`None`)
        method (str, optional): The method to use for negative sampling,
            *i.e.* :obj:`"sparse"` or :obj:`"dense"`.
            This is a memory/runtime trade-off.
            :obj:`"sparse"` will work on any graph of any size, while
            :obj:`"dense"` can perform faster true-negative checks.
            (default: :obj:`"sparse"`)
        force_undirected (bool, optional): If set to :obj:`True`, sampled
            negative edges will be undirected. (default: :obj:`False`)

    :rtype: LongTensor

    Examples:

        >>> # Standard usage
        >>> edge_index = torch.as_tensor([[0, 0, 1, 2], [0, 1, 2, 3]])
        >>> edge_index = torch.cat([edge_index, edge_index + 4], dim=1)
        >>> edge_index
        tensor([[0, 0, 1, 2, 4, 4, 5, 6],
                [0, 1, 2, 3, 4, 5, 6, 7]])
        >>> batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        >>> batched_negative_sampling(edge_index, batch)
        tensor([[3, 1, 3, 2, 7, 7, 6, 5],
                [2, 0, 1, 1, 5, 6, 4, 4]])

        >>> # For bipartite graph
        >>> edge_index1 = torch.as_tensor([[0, 0, 1, 1], [0, 1, 2, 3]])
        >>> edge_index2 = edge_index1 + torch.tensor([[2], [4]])
        >>> edge_index3 = edge_index2 + torch.tensor([[2], [4]])
        >>> edge_index = torch.cat([edge_index1, edge_index2,
        ...                         edge_index3], dim=1)
        >>> edge_index
        tensor([[ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
                [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]])
        >>> src_batch = torch.tensor([0, 0, 1, 1, 2, 2])
        >>> dst_batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        >>> batched_negative_sampling(edge_index,
        ...                           (src_batch, dst_batch))
        tensor([[ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
                [ 2,  3,  0,  1,  6,  7,  4,  5, 10, 11,  8,  9]])
    """
    if isinstance(batch, Tensor):
        src_batch, dst_batch = batch, batch
    else:
        src_batch, dst_batch = batch[0], batch[1]

    split = degree(src_batch[edge_index[0]], dtype=torch.long).tolist()
    edge_indices = torch.split(edge_index, split, dim=1)

    num_src = degree(src_batch, dtype=torch.long)
    cum_src = cumsum(num_src)[:-1]

    if isinstance(batch, Tensor):
        num_nodes = num_src.tolist()
        ptr = cum_src
    else:
        num_dst = degree(dst_batch, dtype=torch.long)
        cum_dst = cumsum(num_dst)[:-1]

        num_nodes = torch.stack([num_src, num_dst], dim=1).tolist()
        ptr = torch.stack([cum_src, cum_dst], dim=1).unsqueeze(-1)

    neg_edge_indices = []
    for i, edge_index in enumerate(edge_indices):
        edge_index = edge_index - ptr[i]
        neg_edge_index = negative_sampling(
            edge_index, num_nodes[i], num_neg_samples, method, force_undirected
        )
        neg_edge_index += ptr[i]
        neg_edge_indices.append(neg_edge_index)

    return torch.cat(neg_edge_indices, dim=1)


def structured_negative_sampling(
    edge_index, num_nodes: Optional[int] = None, contains_neg_self_loops: bool = True
):
    r"""Samples a negative edge :obj:`(i,k)` for every positive edge
    :obj:`(i,j)` in the graph given by :attr:`edge_index`, and returns it as a
    tuple of the form :obj:`(i,j,k)`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        contains_neg_self_loops (bool, optional): If set to
            :obj:`False`, sampled negative edges will not contain self loops.
            (default: :obj:`True`)

    :rtype: (LongTensor, LongTensor, LongTensor)

    Example:

        >>> edge_index = torch.as_tensor([[0, 0, 1, 2],
        ...                               [0, 1, 2, 3]])
        >>> structured_negative_sampling(edge_index)
        (tensor([0, 0, 1, 2]), tensor([0, 1, 2, 3]), tensor([2, 3, 0, 2]))

    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index.cpu()
    pos_idx = row * num_nodes + col
    if not contains_neg_self_loops:
        loop_idx = torch.arange(num_nodes) * (num_nodes + 1)
        pos_idx = torch.cat([pos_idx, loop_idx], dim=0)

    rand = torch.randint(num_nodes, (row.size(0),), dtype=torch.long)
    neg_idx = row * num_nodes + rand

    mask = torch.from_numpy(np.isin(neg_idx, pos_idx)).to(torch.bool)
    rest = mask.nonzero(as_tuple=False).view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.randint(num_nodes, (rest.size(0),), dtype=torch.long)
        rand[rest] = tmp
        neg_idx = row[rest] * num_nodes + tmp

        mask = torch.from_numpy(np.isin(neg_idx, pos_idx)).to(torch.bool)
        rest = rest[mask]

    return edge_index[0], edge_index[1], rand.to(edge_index.device)


def structured_negative_sampling_feasible(
    edge_index: Tensor,
    num_nodes: Optional[int] = None,
    contains_neg_self_loops: bool = True,
) -> bool:
    r"""Returns :obj:`True` if
    :meth:`~torch_geometric.utils.structured_negative_sampling` is feasible
    on the graph given by :obj:`edge_index`.
    :meth:`~torch_geometric.utils.structured_negative_sampling` is infeasible
    if atleast one node is connected to all other nodes.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        contains_neg_self_loops (bool, optional): If set to
            :obj:`False`, sampled negative edges will not contain self loops.
            (default: :obj:`True`)

    :rtype: bool

    Examples:

        >>> edge_index = torch.LongTensor([[0, 0, 1, 1, 2, 2, 2],
        ...                                [1, 2, 0, 2, 0, 1, 1]])
        >>> structured_negative_sampling_feasible(edge_index, 3, False)
        False

        >>> structured_negative_sampling_feasible(edge_index, 3, True)
        True
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    max_num_neighbors = num_nodes

    edge_index = coalesce(edge_index, num_nodes=num_nodes)

    if not contains_neg_self_loops:
        edge_index, _ = remove_self_loops(edge_index)
        max_num_neighbors -= 1  # Reduce number of valid neighbors

    deg = degree(edge_index[0], num_nodes)
    # True if there exists no node that is connected to all other nodes.
    return bool(torch.all(deg < max_num_neighbors))


###############################################################################


def sample(population: int, k: int, device=None) -> Tensor:
    if population <= k:
        return torch.arange(population, device=device)
    else:
        return torch.tensor(random.sample(range(population), k), device=device)


def edge_index_to_vector(
    edge_index: Tensor,
    size: Tuple[int, int],
    bipartite: bool,
    force_undirected: bool = False,
) -> Tuple[Tensor, int]:

    row, col = edge_index

    if bipartite:  # No need to account for self-loops.
        idx = (row * size[1]).add_(col)
        population = size[0] * size[1]
        return idx, population

    elif force_undirected:
        assert size[0] == size[1]
        num_nodes = size[0]

        # We only operate on the upper triangular matrix:
        mask = row < col
        row, col = row[mask], col[mask]
        offset = torch.arange(1, num_nodes, device=row.device).cumsum(0)[row]
        idx = row.mul_(num_nodes).add_(col).sub_(offset)
        population = (num_nodes * (num_nodes + 1)) // 2 - num_nodes
        return idx, population

    else:
        assert size[0] == size[1]
        num_nodes = size[0]

        # We remove self-loops as we do not want to take them into account
        # when sampling negative values.
        mask = row != col
        row, col = row[mask], col[mask]
        col[row < col] -= 1
        idx = row.mul_(num_nodes - 1).add_(col)
        population = num_nodes * num_nodes - num_nodes
        return idx, population


def vector_to_edge_index(
    idx: Tensor, size: Tuple[int, int], bipartite: bool, force_undirected: bool = False
) -> Tensor:

    if bipartite:  # No need to account for self-loops.
        row = idx.div(size[1], rounding_mode="floor")
        col = idx % size[1]
        return torch.stack([row, col], dim=0)

    elif force_undirected:
        assert size[0] == size[1]
        num_nodes = size[0]

        offset = torch.arange(1, num_nodes, device=idx.device).cumsum(0)
        end = torch.arange(
            num_nodes, num_nodes * num_nodes, num_nodes, device=idx.device
        )
        row = torch.bucketize(idx, end.sub_(offset), right=True)
        col = offset[row].add_(idx) % num_nodes
        return torch.stack([torch.cat([row, col]), torch.cat([col, row])], 0)

    else:
        assert size[0] == size[1]
        num_nodes = size[0]

        row = idx.div(num_nodes - 1, rounding_mode="floor")
        col = idx % (num_nodes - 1)
        col[row <= col] += 1
        return torch.stack([row, col], dim=0)
