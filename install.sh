#!/bin/bash

mamba env create -f environment.yml
mamba activate persona-performance
python -m ipykernel install --user --name persona-performance
