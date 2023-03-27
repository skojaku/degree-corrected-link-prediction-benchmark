#!/bin/zsh
conda create -n $1 -c bioconda -c conda-forge python=3.7 ipykernel snakemake snakefmt flake8 pyflakes pep8 pylint jedi pre-commit
pre-commit install

