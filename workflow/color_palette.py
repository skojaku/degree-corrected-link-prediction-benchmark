# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-07-11 22:08:08
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-09 15:12:14
import seaborn as sns
import matplotlib.pyplot as plt


def get_model_order():
    return [
        "line",
        "node2vec",
        "deepwalk",
        "leigenmap",
        "modspec",
        "nonbacktracking",
        "fastrp",
        "SGTLaplacianExp",
        "SGTLaplacianNeumann",
        "SGTAdjacencyExp",
        "SGTAdjacencyNeumann",
        "SGTNormAdjacencyExp",
        "SGTNormAdjacencyNeumann",
        "dcSBM",
        "GCN",
        "GIN",
        "EdgeCNN",
        "GraphSAGE",
        "GAT",
        "dcGCN",
        "dcGIN",
        "dcEdgeCNN",
        "dcGraphSAGE",
        "dcGAT",
    ]


def get_model_names():
    return {
        "line": "LINE",
        "node2vec": "Node2Vec",
        "deepwalk": "DeepWalk",
        "leigenmap": "Laplacian Eigenmap",
        "modspec": "Modularity Spectral",
        "nonbacktracking": "Non-Backtracking",
        "fastrp": "FastRP",
        "SGTLaplacianExp": "SGT Laplacian Exp.",
        "SGTLaplacianNeumann": "SGT Laplacian Neumann.",
        "SGTAdjacencyExp": "SGT Adjacency Exp.",
        "SGTAdjacencyNeumann": "SGT Adjacency Neumann.",
        "SGTNormAdjacencyExp": "SGT Norm Adjacency Exp.",
        "SGTNormAdjacencyNeumann": "SGT Norm Adjacency Neumann.",
        "dcSBM": "Degree-corrected SBM",
        "GCN": "GCN",
        "GIN": "GIN",
        "PNA": "PNA",
        "EdgeCNN": "EdgeCNN",
        "GraphSAGE": "GraphSAGE",
        "GAT": "GAT",
        "dcGCN": "Degree-corrected GCN",
        "dcGIN": "Degree-corrected GIN",
        "dcEdgeCNN": "Degree-corrected EdgeCNN",
        "dcGraphSAGE": "Degree-corrected GraphSAGE",
        "dcGAT": "Degree-corrected GAT",
    }


def get_model_colors():
    cmap = sns.color_palette().as_hex()
    bcmap = sns.color_palette().as_hex()
    mcmap = sns.color_palette("colorblind").as_hex()

    neural_emb_color = bcmap[3]
    spec_emb_color = bcmap[0]
    com_color = bcmap[1]
    neural_emb_color_2 = bcmap[2]
    return {
        "node2vec": "red",
        "deepwalk": sns.desaturate(neural_emb_color, 0.8),
        "line": sns.desaturate(neural_emb_color, 0.2),
        "leigenmap": "#c2c1f1",
        "modspec": sns.desaturate(spec_emb_color, 0.8),
        "nonbacktracking": "blue",
        "fastrp": "green",
        "SGTLaplacianExp": sns.desaturate(com_color, 0.8),
        "SGTLaplacianNeumann": sns.desaturate(com_color, 0.2),
        "SGTAdjacencyExp": sns.desaturate(com_color, 0.5),
        "SGTAdjacencyNeumann": sns.desaturate(com_color, 0.3),
        "SGTNormAdjacencyExp": sns.desaturate(com_color, 0.6),
        "SGTNormAdjacencyNeumann": sns.desaturate(com_color, 0.4),
        "dcSBM": "#aabbcc",
        "GCN": "purple",
        "GIN": "orange",
        "PNA": "yellow",
        "EdgeCNN": "cyan",
        "GraphSAGE": "magenta",
        "GAT": "brown",
        "dcGCN": sns.desaturate("purple", 0.5),
        "dcGIN": sns.desaturate("orange", 0.5),
        "dcEdgeCNN": sns.desaturate("cyan", 0.5),
        "dcGraphSAGE": sns.desaturate("magenta", 0.5),
        "dcGAT": sns.desaturate("brown", 0.5),
    }


def get_model_edge_colors():
    white_color = "white"
    return {
        "node2vec": "black",
        "deepwalk": white_color,
        "line": white_color,
        "leigenmap": "k",
        "modspec": white_color,
        "nonbacktracking": "black",
        "fastrp": "black",
        "SGTLaplacianExp": "k",
        "SGTLaplacianNeumann": "k",
        "SGTAdjacencyExp": "k",
        "SGTAdjacencyNeumann": "k",
        "SGTNormAdjacencyExp": "k",
        "SGTNormAdjacencyNeumann": "k",
        "dcSBM": "k",
        "GCN": "black",
        "GIN": "black",
        "PNA": "black",
        "EdgeCNN": "black",
        "GraphSAGE": "black",
        "GAT": "black",
        "dcGCN": "black",
        "dcGIN": "black",
        "dcEdgeCNN": "black",
        "dcGraphSAGE": "black",
        "dcGAT": "black",
    }


def get_model_linestyles():
    return {
        "line": (2, 2),
        "node2vec": (1, 0),
        "deepwalk": (1, 1),
        "leigenmap": (2, 2),
        "modspec": (1, 1),
        "nonbacktracking": (1, 0),
        "fastrp": (1, 1),
        "SGTLaplacianExp": (2, 2),
        "SGTLaplacianNeumann": (1, 2),
        "SGTAdjacencyExp": (1, 1),
        "SGTAdjacencyNeumann": (2, 2),
        "SGTNormAdjacencyExp": (1, 0),
        "SGTNormAdjacencyNeumann": (1, 1),
        "dcSBM": (2, 2),
        "GCN": (1, 0),
        "GIN": (1, 1),
        "PNA": (2, 2),
        "EdgeCNN": (1, 0),
        "GraphSAGE": (1, 1),
        "GAT": (2, 2),
        "dcGCN": (1, 0),
        "dcGIN": (1, 1),
        "dcEdgeCNN": (2, 2),
        "dcGraphSAGE": (1, 0),
        "dcGAT": (1, 1),
    }


def get_model_markers():
    return {
        "line": "s",
        "node2vec": "s",
        "deepwalk": "s",
        "leigenmap": "o",
        "modspec": "o",
        "nonbacktracking": "o",
        "fastrp": "D",
        "SGTLaplacianExp": "o",
        "SGTLaplacianNeumann": "o",
        "SGTAdjacencyExp": "o",
        "SGTAdjacencyNeumann": "o",
        "SGTNormAdjacencyExp": "o",
        "SGTNormAdjacencyNeumann": "o",
        "dcSBM": "v",
        "GCN": "s",
        "GIN": "s",
        "PNA": "s",
        "EdgeCNN": "s",
        "GraphSAGE": "s",
        "GAT": "s",
        "dcGCN": "o",
        "dcGIN": "o",
        "dcEdgeCNN": "o",
        "dcGraphSAGE": "o",
        "dcGAT": "o",
    }


def get_model_marker_size():
    return {
        "node2vec": 10,
        "line": 10,
        "deepwalk": 10,
        "leigenmap": 10,
        "modspec": 10,
        "nonbacktracking": 10,
        "fastrp": 10,
        "SGTLaplacianExp": 10,
        "SGTLaplacianNeumann": 10,
        "SGTAdjacencyExp": 10,
        "SGTAdjacencyNeumann": 10,
        "SGTNormAdjacencyExp": 10,
        "SGTNormAdjacencyNeumann": 10,
        "dcSBM": 10,
        "GCN": 10,
        "GIN": 10,
        "PNA": 10,
        "EdgeCNN": 10,
        "GraphSAGE": 10,
        "GAT": 10,
        "dcGCN": 10,
        "dcGIN": 10,
        "dcEdgeCNN": 10,
        "dcGraphSAGE": 10,
        "dcGAT": 10,
    }


def get_model_groups():
    return {
        "node2vec": "neural",
        "line": "neural",
        "deepwalk": "neural",
        "leigenmap": "spectral",
        "modspec": "spectral",
        "nonbacktracking": "spectral",
        "fastrp": "neural",
        "SGTLaplacianExp": "spectral",
        "SGTLaplacianNeumann": "spectral",
        "SGTAdjacencyExp": "spectral",
        "SGTAdjacencyNeumann": "spectral",
        "SGTNormAdjacencyExp": "spectral",
        "SGTNormAdjacencyNeumann": "spectral",
        "dcSBM": "spectral",
        "GCN": "neural",
        "GIN": "neural",
        "PNA": "neural",
        "EdgeCNN": "neural",
        "GraphSAGE": "neural",
        "GAT": "neural",
        "dcGCN": "neural",
        "dcGIN": "neural",
        "dcEdgeCNN": "neural",
        "dcGraphSAGE": "neural",
        "dcGAT": "neural",
    }
