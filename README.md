# link-prediction

## Installation

## Create a virtual env

```bash
mamba create -n linkpred -c bioconda -c nvidia -c pytorch -c pyg python=3.11 cuda-version=12.1 pytorch torchvision torchaudio pytorch-cuda=12.1 snakemake graph-tool scikit-learn numpy==1.23.5 numba scipy==1.10.1 pandas polars networkx seaborn matplotlib gensim ipykernel tqdm black faiss-gpu pyg pytorch-sparse python-igraph -y
pip install adabelief-pytorch==0.2.0
pip install GPUtil powerlaw
```

## Install the custom package

```bash
# Embcom package
cd libs/embcom && pip install -e .

# GNN tools
cd libs/gnn-tools && pip install -e .

# LFR benchmark
cd libs/LFR-benchmark && python setup.py build && pip install -e .
```

Create `workflow/config.yaml` file
```config.yaml
data_dir: "data/"
paper_dir: "./paper/current/"
small_networks: False
```

# TODO

- [x] Fix the bug (duplicated positive/negative edges)
- [x] Regenerate the figures
- [x] Create a scatter plot of data vs aucroc ratio (where the preferential attachment is the base) for the uniform sampling
- [x] Create a scatter plot of data vs aucroc ratio (where the preferential attachment is the base) for the biased sampling
- [x] Implement & Run the community detection (LFR & Multi partition model)
  - [x] LFR benchamrk implemented
  - [x] Running the LFR benchmark
  - [x] Collect the results
- [x] Evaluate the community detection benchmark

# Missing module


- Install scipy version 1.10.1.


