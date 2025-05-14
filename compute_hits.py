import pandas as pd
from data.loader import load_data, process_dataset
from data.constants import DATASETS
from utils.answer_extraction import extract_answer, extract_gsm8k, extract_math
from tqdm.auto import tqdm
from pathlib import Path
import glob
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse

parser = argparse.ArgumentParser(
    description='Compute prediction correctness.')
parser.add_argument('--prefix', help='Prefix for path to generation files', type=str, default="./results/")
parser.add_argument('--cpus', help='Number of cpus', type=int, default=None)
args = parser.parse_args()
prefix = args.prefix
if args.cpus is not None:
    cpus_reserved = args.cpus
else:
    cpus_reserved = cpu_count()
print(f"Available CPUs: {cpus_reserved}")

def get_hits(file, dataset, processed_data):
    model = file.split("/")[-3]
    persona = file.split("/")[-1][:-4].replace("_", " ")
    print(f"Computing metrics for {model} and {persona}" + 20*" ", end='\r')
    hits_path = Path(f"{prefix}/hits/{model}/{dataset}/{persona}.csv")
    if hits_path.exists():
        hits_df = pd.read_csv(hits_path)
    else:
        hits_path.parent.mkdir(exist_ok=True, parents=True)
        df = pd.read_csv(file, keep_default_na=False)
        if dataset not in ["gsm8k", "math"]:
            df["n_choices"] = n_choices
            df["answer"] = df.apply(lambda x: extract_answer(x.iloc[0], x.n_choices), axis=1)
        else:
            df["label"] = processed_data["label"]
            if dataset == "gsm8k": df["answer"] = df.apply(lambda x: extract_gsm8k(x.iloc[0]), axis=1)
            else: df["answer"] = df.apply(lambda x: extract_math(x.iloc[0]), axis=1)
            hits = df["label"] == df["answer"]
        if dataset == "truthfulqa":
            df["label"] = processed_data["label"]
            hits = df["label"] == df["answer"]
        elif dataset == "mmlu_pro":
            df["label"] = processed_data["answer"]
            df["category"] = processed_data["category"]
            hits = df["label"] == df["answer"]
        elif dataset == "bigbench":
            df["label"] = processed_data["label"]
            df["category"] = processed_data["category"]
            hits = df["label"] == df["answer"]
        hits_df = pd.DataFrame({model: hits})
        hits_df.to_csv(hits_path, index=False)
    if dataset in ["truthfulqa", "gsm8k"]:
        result = hits_df[model].mean()
    elif dataset in ["mmlu_pro", "bigbench", "math"]:
        hits_df["category"] = processed_data["category"] if dataset != "math" else processed_data["type"]
        result = hits_df.groupby("category")[model].mean().mean()
    return (persona, model, result)

for dataset in DATASETS:
    print(f"Processing {dataset}")
    print()
    data = load_data(dataset)
    if dataset == "truthfulqa":
        n_choices = [len(x["choices"]) for x in data["mc1_targets"]]
    elif dataset == "mmlu_pro":
        n_choices = 10
    elif dataset == "bigbench":
        n_choices = [min(len(x), 26) for x in data["multiple_choice_targets"]]

    processed_data = process_dataset(data, dataset)
    results = {}
    files =  glob.glob(f"{prefix}/*/{dataset}/*")
    func = partial(get_hits, dataset=dataset, processed_data=processed_data)
    pool = Pool(cpus_reserved - 1)
    returns = []
    for r in tqdm(pool.imap_unordered(func, files), total=len(files)):
        returns.append(r)
    # returns = pool.map(func, files)
    pool.close()
    pool.join()
    for persona, model, result in returns:
        results.setdefault(persona, {})[model] = result
    df = pd.DataFrame.from_dict(results, orient="index")
    df.to_csv(f"{prefix}/{dataset}.csv")
