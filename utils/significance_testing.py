from utils.seeds import initialize_seeds
from scipy.stats import binomtest
import pandas as pd
import random
import numpy as np
import scipy


def bootstrap(hits, groupby=None, n_trials=10000):
    sample_means = []
    initialize_seeds()
    
    # loop n_samples times
    for i in range(n_trials):
        if groupby is not None:
            df_bootstrap_sample = hits.groupby(groupby).sample(frac =1., replace = True)
            sample_mean = df_bootstrap_sample.groupby(groupby).mean().mean()
        else:
            df_bootstrap_sample = hits.sample(n = len(hits), replace = True)
            sample_mean = df_bootstrap_sample.mean()
        
        # add this sample mean to the sample means list
        sample_means.append(sample_mean)
    sample_means = pd.DataFrame(sample_means)
    # get the lower bound of the confidence interval
    ci_lower = sample_means.quantile(q = 0.025, numeric_only=True)

    # get the upper bound of the confidence interval
    ci_higher = sample_means.quantile(q = 0.975, numeric_only=True)
    
    return ci_lower, ci_higher


def bootstrap_diff(hits_a, hits_b, groupby=None, n_trials=10000):
    sample_means = []
    initialize_seeds()
    # loop n_samples times
    for i in range(n_trials):
        if groupby is not None:
            df_bootstrap_sample = hits_a.groupby(groupby).sample(frac =1., replace = True)
            sample_mean_a = df_bootstrap_sample.groupby(groupby).mean().mean()
            sample_mean_b = hits_b.loc[df_bootstrap_sample.index].groupby(groupby).mean().mean()
        else:
            df_bootstrap_sample = hits_a.sample(n = len(hits_a), replace = True)
            sample_mean_a = df_bootstrap_sample.mean()
            sample_mean_b = hits_b.loc[df_bootstrap_sample.index].mean()
        
        # add this sample mean to the sample means list
        sample_means.append(sample_mean_a-sample_mean_b)
    sample_means = pd.DataFrame(sample_means)
    # get the lower bound of the confidence interval
    ci_lower = sample_means.quantile(q = 0.025, numeric_only=True).values[0]

    # get the upper bound of the confidence interval
    ci_higher = sample_means.quantile(q = 0.975, numeric_only=True).values[0]
    # print(f"Confidence interval: {ci_lower},{ci_higher}")
    return ci_lower, ci_higher

def bootstrap_fidelity(group_hits, groupby=None, n_trials=10000):
    sample_means = []
    initialize_seeds()
    # loop n_samples times
    for i in range(n_trials):
        bootstraped_scores = []
        if groupby is not None:
            df_bootstrap_sample = group_hits[0].groupby(groupby).sample(frac =1., replace = True)
            for hits in group_hits:
                bootstraped_scores.append(hits.loc[df_bootstrap_sample.index].groupby(groupby).mean().mean().values)
        else:
            df_bootstrap_sample = group_hits[0].sample(n = len(group_hits[0]), replace = True)
            for hits in group_hits:
                bootstraped_scores.append(hits.loc[df_bootstrap_sample.index].mean().values)
        
        # add this sample mean to the sample means list
        df = pd.DataFrame.from_dict({"scores": bootstraped_scores})
        m = scipy.stats.kendalltau(df.sort_values("scores").values, df["scores"].values).statistic
        sample_means.append(m)

    sample_means = pd.DataFrame(sample_means)
    metric = sample_means.mean().values[0]
    # get the lower bound of the confidence interval
    ci_lower = sample_means.quantile(q = 0.025, numeric_only=True).values[0]

    # get the upper bound of the confidence interval
    ci_higher = sample_means.quantile(q = 0.975, numeric_only=True).values[0]
    # print(f"Confidence interval: {ci_lower},{ci_higher}")

    return metric, (ci_lower,ci_higher)

def bootstrap_fidelity_diffs(group_hits, base_group_hits, groupby=None, n_trials=10000):
    sample_means = []
    initialize_seeds()
    # loop n_samples times
    for i in range(n_trials):
        bootstraped_scores = []
        base_bootstraped_scores = []
        if groupby is not None:
            df_bootstrap_sample = group_hits[0].groupby(groupby).sample(frac =1., replace = True)
            for hits in group_hits:
                bootstraped_scores.append(hits.loc[df_bootstrap_sample.index].groupby(groupby).mean().mean().values)
            for hits in base_group_hits:
                base_bootstraped_scores.append(hits.loc[df_bootstrap_sample.index].groupby(groupby).mean().mean().values)
        else:
            df_bootstrap_sample = group_hits[0].sample(n = len(group_hits[0]), replace = True)
            for hits in group_hits:
                bootstraped_scores.append(hits.loc[df_bootstrap_sample.index].mean().values)
            for hits in base_group_hits:
                base_bootstraped_scores.append(hits.loc[df_bootstrap_sample.index].mean().values)
        
        # add this sample mean to the sample means list
        df = pd.DataFrame.from_dict({"scores": bootstraped_scores})
        base_df = pd.DataFrame.from_dict({"scores": base_bootstraped_scores})
        m = scipy.stats.kendalltau(df.sort_values("scores").values, df["scores"].values).statistic
        base_m = scipy.stats.kendalltau(base_df.sort_values("scores").values, base_df["scores"].values).statistic
        sample_means.append(m-base_m)

    sample_means = pd.DataFrame(sample_means)
    metric = sample_means.mean().values[0]
    # get the lower bound of the confidence interval
    ci_lower = sample_means.quantile(q = 0.025, numeric_only=True).values[0]

    # get the upper bound of the confidence interval
    ci_higher = sample_means.quantile(q = 0.975, numeric_only=True).values[0]
    # print(f"Confidence interval: {ci_lower},{ci_higher}")

    return metric, (ci_lower,ci_higher)


def randomized_test(hits_a, hits_b, groupby=None, n_trials=10000):
    if groupby is not None:
        score_a = hits_a.groupby(groupby).mean().mean()
        score_b = hits_b.groupby(groupby).mean().mean()
    else:
        score_a = hits_a.mean()
        score_b = hits_b.mean()
    # print(f'# score(hits_a) = {score_a}')
    # print(f'# score(hits_b) = {score_b}')

    diff = abs(score_a - score_b).values[0]
    # print('# abs(diff) = %f' % diff)
    uncommon = pd.Series(hits_a.index[hits_a.iloc[:,0] != hits_b.iloc[:,0]])

    better = 0
    
    rng = random.Random(42)
    getrandbits_func = rng.getrandbits

    for _ in range(n_trials):   
        hits_a_local, hits_b_local = hits_a.copy(), hits_b.copy()
        flips = []
        for i in uncommon:
            if getrandbits_func(1) == 1:
                flips.append(i)
        hits_a_local.iloc[flips,0], hits_b_local.iloc[flips,0] = hits_b.iloc[flips,0], hits_a.iloc[flips,0]

        assert len(hits_a_local) == len(hits_b_local) == len(hits_a) == len(hits_b)
        if groupby is not None:
            score_a_local = hits_a_local.groupby(groupby).mean().mean()
            score_b_local = hits_b_local.groupby(groupby).mean().mean()
        else:
            score_a_local = hits_a_local.mean()
            score_b_local = hits_b_local.mean()
        diff_local = abs(score_a_local - score_b_local).values[0]
        
        if diff_local >= diff:
            better += 1

    p = (better + 1) / (n_trials + 1)
    # print(f"p-value: {p}")
    return p
    

def get_significance(corrects1, corrects2):
    diffs = corrects2 - corrects1
    successes = (diffs == 1).sum()
    trials = np.abs(diffs).sum()
    if trials == 0: return 1
    return binomtest(successes, trials).pvalue