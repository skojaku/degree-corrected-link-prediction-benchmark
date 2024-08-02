# %%
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.data import DataLoader
import numpy as np
import pandas as pd
import shutil
import os


def prep_and_save(edges, filename):
    _, edges = np.unique(edges.reshape(-1), return_inverse=True)
    edges = edges.reshape(-1, 2)
    edges = edges - np.min(edges) + 1  # To start from 1
    src, trg = edges.T

    pd.DataFrame({"src": src, "trg": trg}).to_csv(
        filename, index=False, header=False, sep=" "
    )


def prep_and_save_biokg(dataset):
    dataset_code = "ogbl-biokg"
    focal_entity_list = list(dataset.num_nodes_dict.keys())
    for focal_entity in focal_entity_list:
        edge_index_keys = [
            k
            for k in dataset.edge_index_dict.keys()
            if k[0] == k[2] and k[0] == focal_entity
        ]
        if len(edge_index_keys) == 0:
            continue

        edges = [dataset.edge_index_dict[k].numpy() for k in edge_index_keys]
        edges = np.concatenate(edges, axis=1)
        prep_and_save(edges, f"net_{dataset_code}_{focal_entity}.txt")


tmp_folder_path = "tmp_dataset/"

small_data = ["ogbl-collab", "ogbl-ddi", "ogbl-biokg"]
medium_data = ["ogbl-citation2", "ogbl-wikikg2", "ogbl-vessel"]
# medium_data = ["ogbl-ppa", "ogbl-citation2", "ogbl-wikikg2", "ogbl-vessel"]
for dataset_code in medium_data:
    # for dataset_code in small_data:

    # Load dataset
    dataset = PygLinkPropPredDataset(name=dataset_code, root=tmp_folder_path)

    # Remove tmp folder
    if os.path.exists(tmp_folder_path):
        shutil.rmtree(tmp_folder_path)

    # Save biokg
    if dataset_code == "ogbl-biokg":
        prep_and_save_biokg(dataset)
        continue

    # Save other datasets
    edges = dataset.edge_index.numpy().T
    prep_and_save(edges, f"net_{dataset_code}.txt")
# %%
