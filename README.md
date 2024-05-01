# link-prediction

## Installation

## Create a virtual env

```bash
mamba create -n linkpred -c bioconda -c nvidia -c pytorch -c pyg python=3.11 cuda-version=12.1 pytorch torchvision torchaudio pytorch-cuda=12.1 snakemake graph-tool scikit-learn numpy numba scipy==1.10.1 pandas polars networkx seaborn matplotlib gensim ipykernel tqdm black faiss-gpu pyg pytorch-sparse python-igraph -y
pip install adabelief-pytorch==0.2.0
pip install GPUtil
```

## Install the custom package

```bash
cd libs/embcom && pip install -e .
cd libs/gnn-tools && pip install -e .
cd libs/linkpred && pip install -e .
```


# TODO

- [x] Fix the bug (duplicated positive/negative edges)
- [ ] Regenerate the figures
- [ ] Create a scatter plot of data vs aucroc ratio (where the preferential attachment is the base) for the uniform sampling
- [ ] Create a scatter plot of data vs aucroc ratio (where the preferential attachment is the base) for the biased sampling
- [ ] Implement & Run the community detection (LFR & Multi partition model)
- [ ] Evaluate the community detection benchmark

# Missing module


- Install scipy version 1.10.1.


#

Satellite aim

The aim of this satellite event is to explore and understand the rise of advanced data representations, such as geometric and topological representations, that are revolutionizing the way we model and comprehend complex systems.
The practice of studying complex systems can be as complex as the systems themselves. Data about complex systems can be massive, relational, and heterogeneous, spanning numerical data, unstructured text, images, and more. Making sense of complex systems often requires reducing their data to convenient and meaningful representations. These new representations provide a fresh view and opens up new avenues for research and discovery.
We hope to see a wide range of researchers from different fields, from computer science and machine learning, network science to complex systems, come together and discuss the latest developments in the field of data representation and its applications as well as the challenges and opportunities that arise in the study of complex systems.

