# Repository for "Implicit degree bias in the link prediction task"

## Citation
```
@article{aiyappa2024implicit,
  title={Implicit degree bias in the link prediction task},
  author={Rachith Aiyappa and Xin Wang and Munjung Kim and Ozgur Can Seckin and Jisung Yoon and Yong-Yeol Ahn and Sadamori Kojaku},
  journal={arxiv: 2405.14985}
  year={2024}
}
```

## Table of content

- [Degree-corrected link prediction task](#degree-corrected-link-prediction-task)
  - [Installation](#installation)
  - [Usage](#usage)
- [Running your link prediction benchmarks](#running-your-link-prediction-benchmarks)
- [Reproducing the results](#reproducing-the-results)
  - [Data](#data)
  - [Installing the packages](#installing-the-packages)
  - [Running the experiments](#running-the-experiments)
  - [Testing with new network data](#testing-with-new-network-data)


# Degree-corrected link prediction task

This repository provides the code to generate the degree-corrected link prediction task.
## Installation
```bash
pip install "git+https://git@github.com/skojaku/degree-corrected-link-prediction.git#subdirectory=libs/dclinkpred&egg=dclinkpred"
```
or
```bash
git clone https://github.com/skojaku/degree-corrected-link-prediction.git
cd degree-corrected-link-prediction/libs/dclinkpred
pip install -e .
```

## Usage

```python
from dclinkpred import LinkPredictionDataset
import networkx as nx

# Create a karate club graph
G = nx.karate_club_graph()

# While the graph can be networkx object, the adjacency matrix is recommended for the efficiency
G = nx.adjacency_matrix(G)

lpdata = LinkPredictionDataset(
    testEdgeFraction=0.2, # 20% of the edges will be used for testing
    degree_correction=True, # degree correction will be applied
    negatives_per_positive=10, # 10 negative samples will be generated for each positive sample
    allow_duplicatd_negatives=False, # Do not allow duplicate negative edges
)
lpdata.fit(G) # Fit the dataset
train_net, src_test, trg_test, y_test = lpdata.transform() # Transform the dataset

train_net # The network for training
src_test # The source nodes of the test edges
trg_test # The destination nodes of the test edges
y_test # The labels of the test edges, where 1 means positive and 0 means negative
```

# Reproducing the results

We provide all source code and data to reproduce the results in the paper. We tested the workflow under the following environment.
- OS: Ubuntu 20.04
- CUDA: 12.1
- Python: 3.11

All code are provided in the `reproduction/` directory. The expected execution time varies depending on the computational resources. With our machine equipped with 8 NVIDIA V100 GPUs and 64 CPUs, the execution time for the entire workflow, including the robustness analysis, is approximately one week.

## Data

We provide the source of the network data in the edge list format at [FigShare](https://figshare.com/projects/Implicit_degree_bias_in_the_link_prediction_task/205432).
The edge list is a CSV file with 2 columns representing the source and destination nodes of the network.
Download the data and place it in the `reproduction/data/raw` directory.

## Installing the packages

We recommend using Miniforge [mamba](https://github.com/conda-forge/miniforge) to manage the packages.

Specifically, we build the conda environment with the following command.
```bash
mamba create -n linkpred -c bioconda -c nvidia -c pytorch -c pyg python=3.11 cuda-version=12.1 pytorch torchvision torchaudio pytorch-cuda=12.1 snakemake graph-tool scikit-learn numpy==1.23.5 numba scipy==1.10.1 pandas polars networkx seaborn matplotlib gensim ipykernel tqdm black faiss-gpu pyg pytorch-sparse python-igraph -y
pip install adabelief-pytorch==0.2.0
pip install GPUtil powerlaw
```
You can also use the `environment.yml` file to create the conda environment.
```bash
mamba env create -f environment.yml
```

Additionally, we need the following custom packages to run the experiments.
- [gnn_tools](https://github.com/skojaku/gnn-tools) provides the code for generating graph embeddings using the GNNs. We used [the version 1.0](https://github.com/skojaku/gnn-tools/releases/tag/v1.0)
- [embcom](https://github.com/skojaku/embcom) provides supplementary graph embedding methods. We used [the version 1.01](https://github.com/skojaku/embcom/releases/tag/v1.01)
- [LFR-benchmark](https://github.com/skojaku/LFR-benchmark) provides the code for the LFR benchmark. We used [version 1.01](https://github.com/skojaku/LFR-benchmark/releases/tag/v1.01).

These packages can be installed via pip as follows:
```bash
pip install git+https://github.com/skojaku/gnn-tools.git@v1.0
#pip install git+https://github.com/skojaku/embcom.git@v1.01
pip install git+https://github.com/igraph/python-igraph
```
And to install the LFR benchmark package:
```bash
git clone https://github.com/skojaku/LFR-benchmark
cd LFR-benchmark
python setup.py build
pip install -e .
```

And install the embcom package:
```bash
git clone https://github.com/skojaku/embcom
cd embcom/libs/embcom
pip install -e .
```

## Running the experiments

We provide the snakemake file to run the experiments. Before running the snakemake, you must create a `config.yaml` file under the `reproduction/workflow/` directory.
```yaml
data_dir: "data/"
small_networks: Fales
```
where `data_dir` is the directory where all data will is located, and `small_networks` is a boolean value indicating whether to run the experiments for the small networks for testing the code.


Once you have created the `config.yaml` file, move under the `reproduction/` directory and run the snakemake as follows:
```bash
snakemake --cores <number of cores> all
```
or conveniently,
```bash
nohup snakemake --cores <number of cores> all >log &
```
The Snakemake will preprocess the data, run the experiments, and generate the figures in `reproduction/figs/` directory.

## Testing with new network data

New networks can be added to the experiment by adding a new file to the `reproduction/data/raw` directory.
The file should be in the edge list format with 2 columns representing the source and destination nodes of the network, e.g.,
```csv
1 2
1 3
1 4
```
where each row forms an edge between the source and destination nodes, and the node IDs should start from 1.

