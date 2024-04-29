# link-prediction

## Installation

## Create a virtual env

```bash
mamba create -n linkpred -c bioconda -c nvidia -c pytorch -c pyg python=3.11 cuda-version=12.1 pytorch torchvision torchaudio pytorch-cuda=12.1 snakemake graph-tool scikit-learn numpy numba scipy pandas polars networkx seaborn matplotlib gensim ipykernel tqdm black faiss-gpu pyg python-igraph -y
pip install adabelief-pytorch==0.2.0
```

## Install the custom package

```bash
cd libs/embcom && pip install -e .
cd libs/gnn-tools && pip install -e .
cd libs/linkpred && pip install -e .
```