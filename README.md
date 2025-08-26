# Principled Personas
This repo holds source code for the paper "Principle Personas: Defining and Measuring the Intended Effects of Persona Prompting on Task Performance", to be presented at EMNLP 2025.


## Requirement

- [miniforge](https://github.com/conda-forge/miniforge)

## Setting up 

1. Run the snippet below to install all dependencies:

```console
conda env create -f environment.yml
```

## Persona generations
- Generations from all personas for all models and prompting strategies is available in the "results" folder.


## Reproducing the experiments
- Run command below to generate dynamic experts:
```console
python inference "google/gemma-2-27b-it" --expertise
```
- Run command below to generate responses for a given model using the default prompting strategy. Replace MODEL_ID with any Huggingface model identifier (e.g., "google/gemma-2-27b-it")
```console
python inference MODEL_ID 
```
- Run command below to generate responses for a given model using a mitigation strategy. Replace MODEL_ID with any Huggingface model identifier (e.g., "google/gemma-2-27b-it"), and MITIGATION with one of {instruction,refine,refine_basic}
```console
python inference MODEL_ID --mitigate MITIGATION
```
- To generate responses for dynamic experts, add --from_expertise. For example, to generate dynamic expert responses using the Refine + Instruction strategy:
```console
python inference MODEL_ID --mitigate refine --from_expertise
```
- To compute and save accuracies for all settings (models, tasks, prompting strategies and personas):
```console
bash ./scripts/save_scores.sh
```
- To compute and save expertise advantage, robustness and fidelity metrics (and pvalues) for all setting (models, tasks, prompting strategies):
```console
python compute_metrics_and_significances.py
```
- Notebook evaluate.ipynb reproduces analysis and figures in the paper.
- Notebook generate_aggregate_images.ipynb reproduces analyses and figures for the aggregate analyzes in the paper.
