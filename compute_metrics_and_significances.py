import pandas as pd
import numpy as np
from data.personas import *
from data.constants import MODEL_ORDER
from data.loader import load_data
from tqdm.auto import tqdm
from utils.significance_testing import *
from utils.metrics import *
from multiprocessing import Pool, cpu_count
import pickle

import argparse

parser = argparse.ArgumentParser(
    description='Compute metrics and significance of results.')
parser.add_argument('--cpus', help='Number of cpus', type=int, default=None)
args = parser.parse_args()
if args.cpus is not None:
    cpus_reserved = args.cpus
else:
    cpus_reserved = cpu_count()
print(f"Available CPUs: {cpus_reserved}")


dataset_dfs = {}
task2persona = {}
persona2task = {}
all_categories = {}

dataset_order = ["truthfulqa", "gsm8k", "mmlu_pro", "bigbench", "math"]

for dataset in dataset_order:
    print(f"Loading and processing {dataset}.")
    dataset_dfs[dataset] = load_data(dataset).to_pandas()
    if dataset == "mmlu_pro":
        persona2task["mmlu_pro"] = {x: "other" if x == "an expert in miscellaneous fields including international relations, sociology, accounting, and human sexuality" else x.replace("an expert in ", "") for x in EXPERTS["mmlu_pro"]}
        all_categories["mmlu_pro"] = dataset_dfs["mmlu_pro"]["category"]
    elif dataset == "bigbench":
        experts = EXPERTS["bigbench"]
        tasks = ['contextual_parametric_knowledge_conflicts',
                'logic_grid_puzzle',
                'strategyqa',
                'tracking_shuffled_objects']
        persona2task["bigbench"] = {
                                        experts[0]: tasks[1],
                                        experts[1]: tasks[2],
                                        experts[2]: tasks[3],
                                        experts[3]: tasks[0]
                                    }
        all_categories["bigbench"] = dataset_dfs["bigbench"]["category"]
    elif dataset == "math":
        experts = EXPERTS[dataset][1:8]
        tasks =  ['Algebra',
                'Counting & Probability',
                'Geometry',
                'Intermediate Algebra',
                'Number Theory',
                'Prealgebra',
                'Precalculus']
        persona2task["math"] = {p: s for p, s in zip(experts, tasks)}
        all_categories["math"] = dataset_dfs["math"]["type"]

for k, v in persona2task.items():
    task2persona[k] = {value: key for key,value in v.items()}


task_to_dataset = {
    "truthfulqa": "truthfulqa",
    "gsm8k": "gsm8k",
}
mmlu_tasks = {x: "mmlu_pro" for x in task2persona["mmlu_pro"].keys()}
bigbench_tasks = {x: "bigbench" for x in task2persona["bigbench"].keys()}
math_tasks = {x: "math" for x in task2persona["math"].keys()}
task_to_dataset = {**task_to_dataset, **mmlu_tasks, **bigbench_tasks, **math_tasks}

expert_fields = [
 'stem',
 'other',
 'stem',
 'stem',
 'social sciences',
 'stem',
 'other',
 'humanities',
 'humanities',
 'stem',
 'other',
 'humanities',
 'stem',
 'social sciences'
]

mapping = {s: f for s, f in zip(np.unique(list(mmlu_tasks.keys())), expert_fields)}

def task_to_related(task):
    dataset = task_to_dataset[task]
    persona = task2persona[dataset][task]
    if dataset == "mmlu_pro": 
        experts = EXPERTS[dataset]
        return [x for x in experts if x != persona]
    if dataset == "bigbench":
        experts = EXPERTS[dataset][:4]
        return [x for x in experts if x != persona]
    if dataset == "math":
        experts = EXPERTS[dataset][1:8]
        return [x for x in experts if x != persona]
    
def noOverlap(a, b):
    if a[1] < b[0]:
        return True
    elif a[0] > b[1]:
        return True
    else:
        return False

    
def task_to_outs(task):
    dataset = task_to_dataset[task]
    experts = EXPERTS[dataset]
    if dataset == "truthfulqa": return [x for x in experts if "expert" in x and "fact" not in x]
    if dataset == "gsm8k": return [ x for x in experts if "expert" in x and "math" not in x]
    if dataset == "mmlu_pro": 
        field = mapping[task]
        return [x for x in experts if mapping[persona2task["mmlu_pro"][x]] != field]
    if dataset == "bigbench":
        out_experts = EXPERTS[dataset][4:]
    if dataset == "math":
        out_experts = EXPERTS[dataset][8:]
    return out_experts

def expertise_to_personas(task, expertise):
    if expertise == "out-expert":
        return task_to_outs(task)
    elif expertise == "experts":
        return task_to_related(task)
    elif expertise == "in-expert":
        dataset = task_to_dataset[task]
        if dataset == "mmlu_pro": 
            return [task2persona[dataset][task]]
        if dataset == "bigbench":
            return [task2persona[dataset][task]]
        if dataset == "math":
            return [task2persona[dataset][task]]
        return [EXPERTS[dataset][0]]
    
def compute_fidelity(task, df,persona_set, prefix, return_pvalues=True):
    corrs =[]
    significants= []
    intervals = []
    dataset = task_to_dataset[task]
    for col in df.columns:
        try:
            group_hits = [pd.read_csv(f"{prefix}/hits/{col}/{dataset}/{x}.csv") for x in persona_set]
        except FileNotFoundError:
            group_hits = []
            for expertise in persona_set:
                personas = expertise_to_personas(task, expertise)
                expertise_hits = [pd.read_csv(f"{prefix}/hits/{col}/{dataset}/{x}.csv").astype(float) for x in personas]
                group_hits.append(sum(expertise_hits)/len(expertise_hits))
        if dataset not in ["truthfulqa", "gsm8k"]:
            idxs = all_categories[dataset] == task
            group_hits = [x[idxs] for x in group_hits]
        corr, interval = bootstrap_fidelity(group_hits)
        corrs.append(corr)
        significants.append(interval[0]*interval[1]>0)
        intervals.append(interval)
    if return_pvalues:
        return  corrs, intervals, significants
    else:
        return corrs

def compute_metrics(job):
    task, df, metric = job
    print(f"Computing {metric} metric for {task}." + 20*" ", end='\r')
    if metric == "empty":
       metric_value = df.loc["empty"]
    if metric == "OP":
       metric_value = compute_op(df, "in-expert")
    if metric == "Fid_Ed":
       metric_value, interval, significance = compute_fidelity(task, df, EDUCATION_PERSONAS, prefix)
    if metric == "Fid_Exp":
        if task in ["truthfulqa", "gsm8k"]:
            experts = ["out-expert", "in-expert"]
        else:
            experts =  ["out-expert", "experts", "in-expert"]
        metric_value, interval, significance = compute_fidelity(task, df, experts, prefix)
    if metric == "Fid_ExpLevel":
        metric_value, interval, significance  = compute_fidelity(task, df, ["level1", "level2", "level3"], prefix)
    if "Fid" in metric:
        metric_value = (metric_value, interval, significance)
    if metric == "WU_color":
        metric_value = worst_case_utility(df, COLOR_PERSONAS)
    if metric == "WU_name":
        metric_value = worst_case_utility(df, NAMES)
    return task, metric, metric_value

def compute_fidelity_diffs(task, df,persona_set, prefix, return_pvalues=True):
    corrs =[]
    significants= []
    intervals = []
    dataset = task_to_dataset[task]
    for col in df.columns:
        try:
            group_hits = [pd.read_csv(f"{prefix}/hits/{col}/{dataset}/{x}.csv") for x in persona_set]
            base_group_hits =  [pd.read_csv(f"./results/hits/{col}/{dataset}/{x}.csv") for x in persona_set]
        except FileNotFoundError:
            group_hits = []
            base_group_hits = []
            for expertise in persona_set:
                personas = expertise_to_personas(task, expertise)
                expertise_hits = [pd.read_csv(f"{prefix}/hits/{col}/{dataset}/{x}.csv").astype(float) for x in personas]
                base_expertise_hits = [pd.read_csv(f"./results/hits/{col}/{dataset}/{x}.csv").astype(float) for x in personas]
                group_hits.append(sum(expertise_hits)/len(expertise_hits))
                base_group_hits.append(sum(base_expertise_hits)/len(base_expertise_hits))
        if dataset not in ["truthfulqa", "gsm8k"]:
            idxs = all_categories[dataset] == task
            group_hits = [x[idxs] for x in group_hits]
            base_group_hits = [x[idxs] for x in base_group_hits]
        corr, interval = bootstrap_fidelity_diffs(group_hits, base_group_hits)
        corrs.append(corr)
        significants.append(interval[0]*interval[1]>0)
        intervals.append(interval)
    if return_pvalues:
        return  corrs, intervals, significants
    else:
        return corrs

def compute_metrics_diffs(job):
    task, df, metric, base_value= job
    print(f"Computing {metric} metric for {task}." + 20*" ", end='\r')
    if metric == "empty":
       metric_value = df.loc["empty"]
    if metric == "OP":
       metric_value = compute_op(df, "in-expert")
    if metric == "Fid_Ed":
       metric_value, interval, significance = compute_fidelity_diffs(task, df, EDUCATION_PERSONAS, prefix)
    if metric == "Fid_Exp":
        if task in ["truthfulqa", "gsm8k"]:
            experts = ["out-expert", "in-expert"]
        else:
            experts =  ["out-expert", "experts", "in-expert"]
        metric_value, interval, significance = compute_fidelity_diffs(task, df, experts, prefix)
    if metric == "Fid_ExpLevel":
        metric_value, interval, significance  = compute_fidelity_diffs(task, df, ["level1", "level2", "level3"], prefix)
    if metric == "WU_color":
        metric_value = worst_case_utility(df, COLOR_PERSONAS)
    if metric == "WU_name":
        metric_value = worst_case_utility(df, NAMES)
    if "Fid" in metric:
        metric_value = (metric_value, interval, significance)
    else:
        metric_value = metric_value - base_value
    return task, metric, metric_value

for prefix in ["./results", "./results/instruction", "./results/refine", "results/refine_basic"]:
    print(f"Processing {prefix}")
    try:
        fidelity_significance = pickle.load(open(f"{prefix}/fidelity_significances.pkl", "rb"))
    except FileNotFoundError:
        fidelity_significance = {}
    try:
        fidelity_intervals = pickle.load(open(f"{prefix}/fidelity_intervals.pkl", "rb"))
    except FileNotFoundError:
        fidelity_intervals = {}
    try:
        all_pvalues = pickle.load(open(f"{prefix}/all_pvalues.pkl", "rb"))
    except FileNotFoundError:
        all_pvalues = {}
    try:
        all_metrics = pickle.load(open(f"{prefix}/all_metrics.pkl", "rb"))
    except FileNotFoundError:
        all_metrics = {}
    try:
        all_results = pickle.load(open(f"{prefix}/all_results.pkl", "rb"))
    except FileNotFoundError:
        all_results = {}
    if prefix != "./results":
        all_results_base = pickle.load(open(f"./results/all_results.pkl", "rb"))
        all_metrics_base = pickle.load(open(f"./results/all_metrics.pkl", "rb"))
        try:
            fidelity_significance_base = pickle.load(open(f"{prefix}/fidelity_significances_base.pkl", "rb"))
        except FileNotFoundError:
            fidelity_significance_base = {}
        try:
            all_pvalues_base = pickle.load(open(f"{prefix}/all_pvalues_base.pkl", "rb"))
        except FileNotFoundError:
            all_pvalues_base = {}
        try:
            all_metrics_diffs = pickle.load(open(f"{prefix}/all_metrics_diffs.pkl", "rb"))
        except FileNotFoundError:
            all_metrics_diffs = {}
        try:
            all_results_diffs = pickle.load(open(f"{prefix}/all_results_diffs.pkl", "rb"))
        except FileNotFoundError:
            all_results_diffs = {}
        try:
            fidelity_intervals_diffs = pickle.load(open(f"{prefix}/fidelity_intervals_diffs.pkl", "rb"))
        except FileNotFoundError:
            fidelity_intervals_diffs = {}

    for dataset in dataset_order:
        print(f"Processing {dataset}.")
        results = pd.read_csv(f"{prefix}/{dataset}.csv", index_col=0)[MODEL_ORDER].dropna()
        if prefix != "./results":
            results.loc["empty"] = pd.read_csv(f"./results/{dataset}.csv", index_col=0)[MODEL_ORDER].loc["empty"]
        df = dataset_dfs[dataset]
        if dataset not in ["mmlu_pro", "bigbench", "math"]:
            if (dataset in all_results.keys() and dataset in all_pvalues.keys()):
                if prefix != "./results" and dataset in all_pvalues_base.keys() and dataset in all_results_diffs.keys():
                    continue
                elif prefix == "./results":
                    continue
            expert = expertise_to_personas(dataset, "in-expert")
            pvalues = {}
            pvalues_base = {}
            for model in tqdm(MODEL_ORDER):
                empty_hits = pd.read_csv(f"./results/hits/{model}/{dataset}/empty.csv")
                for persona in tqdm(results.index):
                    if persona == "empty":
                        hits = empty_hits
                    else:
                        hits = pd.read_csv(f"{prefix}/hits/{model}/{dataset}/{persona}.csv")
                    pvalue = get_significance(hits.iloc[:,0].astype(int).values, empty_hits.iloc[:,0].astype(int).values)
                    pvalues.setdefault(model, {})[persona]=pvalue
                    if prefix != "./results":
                        base_hits = pd.read_csv(f"./results/hits/{model}/{dataset}/{persona}.csv")
                        pvalue_base = get_significance(hits.iloc[:,0].astype(int).values, base_hits.iloc[:,0].astype(int).values)
                        pvalues_base.setdefault(model, {})[persona]=pvalue_base
            results.loc["in-expert"] = results.loc[expert[0]]
            results.loc["out-expert"] = results.loc[task_to_outs(dataset)].mean()
            pvalues =  pd.DataFrame.from_dict(pvalues, orient="index").T
            pvalues.loc["in-expert"] = pvalues.loc[expert[0]]
            all_results[dataset] = results
            all_pvalues[dataset] = pvalues
            if prefix != "./results":
                pvalues_base =  pd.DataFrame.from_dict(pvalues_base, orient="index").T
                pvalues_base.loc["in-expert"] = pvalues_base.loc[expert[0]]
                all_pvalues_base[dataset] = pvalues_base
                base_results = all_results_base[dataset]
                all_results_diffs[dataset] = results - base_results

        else:
            categories = all_categories[dataset]
            scores_per_task = {}
            pvalues_per_task = {}
            pvalues_base_per_task = {}
            for model in tqdm(MODEL_ORDER):
                empty_hits = pd.read_csv(f"./results/hits/{model}/{dataset}/empty.csv")
                empty_hits["category"] = categories
                for persona in tqdm(results.index):
                    if persona == "empty":
                        hits = empty_hits
                    else:
                        hits = pd.read_csv(f"{prefix}/hits/{model}/{dataset}/{persona}.csv")
                    hits["category"] = categories
                    hits_per_task = hits.groupby("category")
                    if prefix != "./results":
                        base_hits =  pd.read_csv(f"./results/hits/{model}/{dataset}/{persona}.csv")
                        base_hits["category"] = categories
                        base_hits_per_task = base_hits.groupby("category")
                    for name, group in hits_per_task:
                        if name in all_results.keys() and name in all_pvalues.keys():
                            if prefix != "./results" and name in all_pvalues_base.keys() and name in all_results_diffs.keys():
                                continue
                            elif prefix == "./results":
                                continue
                        scores_per_task.setdefault(name, {}).setdefault(model, {})[persona]=group.mean(numeric_only=True).values[0]
                        pvalue = get_significance(group.iloc[:,0].astype(int).values, empty_hits[empty_hits.category == name].iloc[:,0].astype(int).values)
                        pvalues_per_task.setdefault(name, {}).setdefault(model, {})[persona]=pvalue
                        if prefix != "./results":
                            pvalue_base = get_significance(group.iloc[:,0].astype(int).values, base_hits[base_hits.category == name].iloc[:,0].astype(int).values)
                            pvalues_base_per_task.setdefault(name, {}).setdefault(model, {})[persona]=pvalue_base
            for task, r in scores_per_task.items():
                scores_per_task[task] = pd.DataFrame.from_dict(r, orient="index").T
                pvalues_per_task[task] =  pd.DataFrame.from_dict(pvalues_per_task[task], orient="index").T
                if prefix != "./results":
                    pvalues_base_per_task[task] =  pd.DataFrame.from_dict(pvalues_base_per_task[task], orient="index").T
            for task in scores_per_task.keys():
                persona = task2persona[task_to_dataset[task]][task]
                scores_per_task[task].loc["in-expert"] =  scores_per_task[task].loc[persona]
                pvalues_per_task[task].loc["in-expert"] =  pvalues_per_task[task].loc[persona]
                scores_per_task[task].loc["experts"] =  scores_per_task[task].loc[task_to_related(task)].mean()
                scores_per_task[task].loc["out-expert"] =  scores_per_task[task].loc[task_to_outs(task)].mean()
                if prefix != "./results":
                    pvalues_base_per_task[task].loc["in-expert"] =  pvalues_base_per_task[task].loc[persona]
                    for task, results in scores_per_task.items():
                        base_results = all_results_base[task]
                        all_results_diffs[task] = results - base_results
                
            all_results = {**all_results, **scores_per_task}
            all_pvalues = {**all_pvalues, **pvalues_per_task}
            if prefix != "./results": all_pvalues_base = {**all_pvalues_base, **pvalues_base_per_task}
    jobs = []
    for task, df in all_results.items():
        for metric in ['empty', 'OP', 'Fid_Ed', 'Fid_Exp', 'Fid_ExpLevel', 'WU_color', 'WU_name']:
            if task not in all_metrics or metric not in all_metrics[task].index:
                jobs.append((task, df, metric))
            elif "Fid" in metric and (task not in fidelity_intervals or metric not in fidelity_intervals[task].index):
                jobs.append((task, df, metric))
    print(f'Metrics to compute: {", ".join([str((j[0], j[2])) for j in jobs])}')
    pool = Pool(cpus_reserved-1)
    returns = []
    for task, metric, metric_value in tqdm(pool.imap_unordered(compute_metrics, jobs), total=len(jobs)):
        if "Fid" in metric:
            metric_value, interval, significance = metric_value
            if task not in fidelity_significance:
                fidelity_significance[task] = pd.DataFrame(columns=MODEL_ORDER)
            fidelity_significance[task].loc[metric] = significance
            if task not in fidelity_intervals:
                fidelity_intervals[task] = pd.DataFrame(columns=MODEL_ORDER)
            fidelity_intervals[task].loc[metric] = interval
        if task not in all_metrics:
            all_metrics[task] = pd.DataFrame(columns=MODEL_ORDER)
        all_metrics[task].loc[metric] = metric_value
    pool.close()
    pool.join()
    for task, df in all_metrics.items():
        all_metrics[task] = df.loc[['empty', 'OP', 'Fid_Ed', 'Fid_Exp', 'Fid_ExpLevel', 'WU_color', 'WU_name']]
        all_results[task] = all_results[task].sort_index()
        all_pvalues[task] = all_pvalues[task].sort_index()
        if prefix != "./results": 
            all_pvalues_base[task] = all_pvalues_base[task].sort_index()
            all_results_diffs[task] = all_results_diffs[task].sort_index()
    if prefix != "./results":
        jobs = []
        for task, df in all_results.items():
            for metric in ['empty', 'OP', 'Fid_Ed', 'Fid_Exp', 'Fid_ExpLevel', 'WU_color', 'WU_name']:
                if task not in all_metrics_diffs or metric not in all_metrics_diffs[task].index:
                    base_value = all_metrics_base[task].loc[metric]
                    jobs.append((task, df, metric, base_value))
                elif "Fid" in metric and (task not in fidelity_intervals_diffs or metric not in fidelity_intervals_diffs[task].index):
                    jobs.append((task, df, metric, None))
        print(f'Metric diffs to compute: {", ".join([str((j[0], j[2])) for j in jobs])}')
        pool = Pool(cpus_reserved-1)
        returns = []
        for task, metric, metric_value in tqdm(pool.imap_unordered(compute_metrics_diffs, jobs), total=len(jobs)):
            if "Fid" in metric:
                metric_value, interval, significance = metric_value
                if task not in fidelity_significance_base:
                    fidelity_significance_base[task] = pd.DataFrame(columns=MODEL_ORDER)
                fidelity_significance_base[task].loc[metric] = significance
                if task not in fidelity_intervals_diffs:
                    fidelity_intervals_diffs[task] = pd.DataFrame(columns=MODEL_ORDER)
                fidelity_intervals_diffs[task].loc[metric] = interval
            if task not in all_metrics_diffs:
                all_metrics_diffs[task] = pd.DataFrame(columns=MODEL_ORDER)
            all_metrics_diffs[task].loc[metric] = metric_value
        pool.close()
        pool.join()
        for task, df in all_metrics_diffs.items():
            all_metrics_diffs[task] = df.loc[['empty', 'OP', 'Fid_Ed', 'Fid_Exp', 'Fid_ExpLevel', 'WU_color', 'WU_name']]
    pickle.dump(fidelity_significance,open(f"{prefix}/fidelity_significances.pkl", "wb"))
    pickle.dump(fidelity_intervals,open(f"{prefix}/fidelity_intervals.pkl", "wb"))
    pickle.dump(all_metrics,open(f"{prefix}/all_metrics.pkl", "wb"))
    pickle.dump(all_results,open(f"{prefix}/all_results.pkl", "wb"))
    pickle.dump(all_pvalues,open(f"{prefix}/all_pvalues.pkl", "wb"))
    if prefix != "./results":
        pickle.dump(all_pvalues_base,open(f"{prefix}/all_pvalues_base.pkl", "wb"))
        pickle.dump(fidelity_significance_base,open(f"{prefix}/fidelity_significances_base.pkl", "wb"))
        pickle.dump(all_metrics_diffs,open(f"{prefix}/all_metrics_diffs.pkl", "wb"))
        pickle.dump(all_results_diffs,open(f"{prefix}/all_results_diffs.pkl", "wb"))
        pickle.dump(fidelity_intervals_diffs,open(f"{prefix}/fidelity_intervals_diffs.pkl", "wb"))
        