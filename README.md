# link-prediction
Link prediction


## Setup

```bash
conda create -n linkpred python=3.9
conda activate linkpred
conda install -c conda-forge mamba -y
mamba install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge -y
mamba install -y -c bioconda -c conda-forge snakemake -y
mamba install -c conda-forge graph-tool scikit-learn numpy numba scipy pandas networkx seaborn matplotlib gensim ipykernel tqdm black -y
mamba install -c conda-forge numpy==1.23.5
```

Install the in-house packages:
```bash 
cd libs/linkpred && pip install -e .
cd libs/embcom && pip install -e .
```
