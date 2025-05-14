#!/usr/bin/env bash

python compute_hits.py
python compute_hits.py --prefix "./results/instruction"
python compute_hits.py --prefix "./results/refine"
python compute_hits.py --prefix "./results/refine_basic"