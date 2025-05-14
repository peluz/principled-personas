import pandas as pd
import scipy

def load_invariances(prefix, metrics, dataset):
    invariances = pd.read_csv(f"{prefix}/invariances/{dataset}.csv", index_col=0)
    return pd.concat([metrics, invariances], axis=0)

def compute_fidelity(df, persona_set, return_pvalues=False):
    corrs =[]
    pvalues= []
    for col in df.columns:
        corr, pvalue = scipy.stats.kendalltau(df.loc[persona_set][col].sort_values().values, df.loc[persona_set][col].values)
        corrs.append(corr)
        pvalues.append(pvalue)
    if return_pvalues:
        return  corrs, pvalues
    else:
        return corrs
    
def compute_op(df, expert):
    # return (df.loc[expert] - df.loc["empty"]) / (1 -df.loc["empty"])
    return df.loc[expert] - df.loc["empty"]

def worst_case_utility(df, persona_set, return_persona= False):
    worst_score = df.loc[persona_set].min()
    # utility = (worst_score - df.loc["empty"]) / (1 -df.loc["empty"])
    utility = worst_score - df.loc["empty"]
    if return_persona:
        persona = df.loc[persona_set].idxmin()
        return utility, persona
    else: return utility