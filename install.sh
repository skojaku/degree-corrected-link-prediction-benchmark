mamba create -n linkpred -c bioconda -c nvidia -c pytorch -c pyg python=3.11 cuda-version=12.1 pytorch torchvision torchaudio pytorch-cuda=12.1 snakemake graph-tool scikit-learn numpy numba scipy pandas polars networkx seaborn matplotlib gensim ipykernel tqdm black faiss-gpu pyg  -y
#pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
#pip install torch_geometric
