"""
This implementation of ABae differs from the one in algorithm.py in that
it is intended to be used in real world situations where all the statistics
and predicates of a dataset are not precomputed.
"""
import numpy as np

def abae(data_records, proxy_scores, oracle_fn, n1, n2, k=5):
    data_records = np.array(data_records)
    proxy_scores = np.array(proxy_scores)

    assert len(data_records) == len(proxy_scores)
    assert len(proxy_scores.shape) == 1

    p_est = np.zeros(k)
    std_est = np.zeros(k)
    resovoir = []

    strata = np.array_split(data_records, k)
    for i in range(k):
        stratum = strata[i]
        sample_idxs = np.random.choice(stratum.shape[0], n1, replace=False)
        samples = stratum[sample_idxs]
        statistics = np.zeros(samples.shape[0]).astype(np.float32)
        predicates = np.zeros(samples.shape[0]).astype(bool)
        for j in range(samples.shape[0]):
            statistic, predicate = oracle_fn(samples[j])
            statistics[j] = statistic
            predicates[j] = predicate
        if samples.shape[0] > 0:
            p_est[i] = predicates.mean()
        if predicates.sum() > 1:
            std_est[i] = np.std(statistics[predicates], ddof=1)
        resovoir.append([statistics, predicates])
    
    weights = np.sqrt(p_est) * std_est
    norm = 1 if np.sum(weights) == 0 else np.sum(weights)
    weights = weights / norm
    allocation = np.floor(n2 * weights).astype(np.int64)

    mu_est = np.zeros(k)
    for i in range(k):
        stratum = strata[i]
        sample_idxs = np.random.choice(stratum.shape[0], allocation[i], replace=False)
        samples = stratum[sample_idxs]
        statistics = np.zeros(samples.shape[0]).astype(np.float32)
        predicates = np.zeros(samples.shape[0]).astype(bool)
        for j in range(samples.shape[0]):
            statistic, predicate = oracle_fn(samples[j])
            statistics[j] = statistic
            predicates[j] = predicate
        resovoir[i][0] = np.concatenate([statistics, resovoir[i][0]])
        resovoir[i][1] = np.concatenate([predicates, resovoir[i][1]])

        if resovoir[i][1].shape[0] > 0:
            p_est[i] = resovoir[i][1].mean()
        if resovoir[i][1].sum() > 0:
            mu_est[i] = resovoir[i][0][resovoir[i][1]].mean()

    norm = 1 if p_est.sum() == 0 else p_est.sum()
    estimate = np.sum(mu_est * p_est / norm)

    ci = bootstrap(resovoir, n1, n2, k)
    return estimate, ci

def bootstrap(resovoir, n1, n2, k, bootstrap_trials=1000, confidence=0.95):
    resamples = np.zeros(bootstrap_trials)
    for b in range(bootstrap_trials):
        p = np.zeros(k)
        s = np.zeros(k)
        bootstrap_resovoir = []
        for j in range(k):
            idxs = np.arange(resovoir[j][0].shape[0])
            sample_idxs = np.random.choice(idxs, n1, replace=True)
            statistics = resovoir[j][0][sample_idxs]
            predicates = resovoir[j][1][sample_idxs]
            bootstrap_resovoir.append([statistics, predicates])
            if len(predicates) > 0:
                p[j] = np.mean(predicates)
            if np.sum(predicates) > 1:
                s[j] = np.std(statistics[predicates], ddof=1)
        
        weights = np.sqrt(p) * s
        norm = 1 if np.sum(weights) == 0 else np.sum(weights)
        weights = weights / norm
        allocation = np.floor(n2 * weights).astype(np.int64)

        m = np.zeros(k)
        for j in range(k):
            idxs = np.arange(resovoir[j][0].shape[0])
            if len(idxs) != 0:
                sample_idxs = np.random.choice(idxs, allocation[j], replace=True)
                statistics = resovoir[j][0][sample_idxs]
                statistics = np.concatenate([statistics, bootstrap_resovoir[j][0]])
                predicates = resovoir[j][1][sample_idxs]
                predicates = np.concatenate([predicates, bootstrap_resovoir[j][1]])
            else:
                statistics = bootstrap_resovoir[j][0]
                predicates = bootstrap_resovoir[j][1]
            if len(predicates) > 0:
                p[j] = np.mean(predicates)
            if np.sum(predicates) > 0:
                m[j] = np.mean(statistics[predicates])
                
        norm = 1 if p.sum() == 0 else p.sum()
        resamples[b] = np.sum(m * p / norm)

    lower_percentile = ((1 - confidence) / 2) * 100
    lower_bound = np.percentile(resamples, lower_percentile)
    upper_percentile = 100 - lower_percentile
    upper_bound = np.percentile(resamples, upper_percentile)
    return lower_bound, upper_bound

if __name__ == "__main__":
    import pandas as pd

    # Sanity Check
    df = pd.read_csv("../data/jackson.csv")
    proxy_scores = df["proxy_scores"].to_numpy()
    statistics = df["statistics"].to_numpy()
    predicates = df["predicates"].to_numpy()
    data_records = [[a, b] for a, b in zip(statistics, predicates)]
    oracle_fn = lambda x: x
    estimate, ci = abae(data_records, proxy_scores, oracle_fn, 2000, 2000)
    print("Estimate:", estimate)
    print("Confidence Interval:", ci)
