import numpy as np
import ray
import mystic as my
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import abae
ray.init(ignore_reinit_error=True)

HOME = abae.data.HOME

blond = np.load(HOME + "celeba-blond.npy", allow_pickle=True).item()
brown = np.load(HOME + "celeba-brown.npy", allow_pickle=True).item()
black = np.load(HOME + "celeba-black.npy", allow_pickle=True).item()
gray = np.load(HOME + "celeba-gray.npy", allow_pickle=True).item()

blond_paths = [path.split("/")[-1] for path in blond["paths"]]
brown_paths = [path.split("/")[-1] for path in brown["paths"]]
black_paths = [path.split("/")[-1] for path in black["paths"]]
gray_paths = [path.split("/")[-1] for path in gray["paths"]]

blond_sort = np.argsort(blond_paths)
brown_sort = np.argsort(brown_paths)
black_sort = np.argsort(black_paths)
gray_sort = np.argsort(gray_paths)

fp = HOME + "list_attr_celeba.txt"
df = pd.read_csv(fp, delim_whitespace=True, header=1)
smiling_statistics = df["Smiling"].replace(-1, 0).to_numpy()


class SingleOracleRecords:
    def __init__(self, k, proxy_scores, statistics, predicates):
        self.k = k
        self.g = len(predicates)
        self.proxy_scores = proxy_scores
        self.statistics = statistics
        self.predicates = predicates
        self.sort = np.argsort(proxy_scores, axis=1)
        self.ground_truth = np.array([self.statistics[self.predicates[g]].mean() for g in range(self.g)])
        
    def pepsi(self, n):
        lgk = np.zeros([n, self.g, self.g, self.k]).astype(bool)
        sample_idxs = np.random.choice(self.sort.shape[1], n, replace=False)
        statistics = self.statistics[sample_idxs]
        for idx, sample_idx in enumerate(sample_idxs):
            for g in range(self.g):
                if not self.predicates[g, sample_idx]:
                    continue
                for l in range(self.g):
                    stratified_idx = np.where(self.sort[l] == sample_idx)[0][0]
                    k = int(stratified_idx // (self.sort.shape[1] / self.k))
                    lgk[idx, l, g, k] = True
        return statistics, lgk
    
    def sample(self, n, l, k=None):
        sorted_statistics = self.statistics[self.sort[l]]
        sorted_predicates = self.predicates[:, self.sort[l]]
        if k is None:
            sample_idxs = np.random.choice(np.arange(self.sort.shape[1]), n, replace=False)
        else:
            stratum = np.array_split(np.arange(self.sort.shape[1]), self.k)[k]
            sample_idxs = np.random.choice(stratum, n, replace=False)
        
        statistics = sorted_statistics[sample_idxs]
        predicates = sorted_predicates[:, sample_idxs]
        return statistics, predicates


proxy_scores = np.array([
     gray["proxy_scores"][gray_sort][:, 0],
     blond["proxy_scores"][blond_sort][:, 0],
])

statistics = smiling_statistics

predicates = np.array([
    np.array(gray["predicates"])[gray_sort],
    np.array(blond["predicates"])[blond_sort]
])

dblgk = SingleOracleRecords(5, proxy_scores, statistics, predicates)


import random


N = 100000
rng = np.random.RandomState(3212142)
proxy_scores_a = np.zeros(N)

for i in range(N):
    if random.randint(0, 100) > 85:
        proxy_scores_a[i] = random.randint(85, 100) / 100

statistics = np.random.normal(15, 5, N)


predicates = np.zeros([4, N])
for i in range(N):
    g = np.random.choice(np.arange(0, 4))
    predicates[g, i] = rng.binomial(n=1, p=proxy_scores_a[i])
    

predicates = predicates.astype(bool)
    
# predicates = np.array([
#     rng.binomial(n=1, p=proxy_scores_a).astype(bool),
#     rng.binomial(n=1, p=proxy_scores_a).astype(bool),
#     rng.binomial(n=1, p=proxy_scores_a).astype(bool),
#     rng.binomial(n=1, p=proxy_scores_a).astype(bool),
# ])

trash = np.random.uniform(0, 1, N)
proxy_scores = np.array([
    proxy_scores_a,
    trash,
    trash,
    trash
])

dblgk = SingleOracleRecords(5, proxy_scores, statistics, predicates)


predicates.sum(axis=1) / len(predicates[0])


def foo(db, n1, n2, version="optimize"):
    L = db.g
    G = db.g
    K = db.k
    ps = np.zeros([L, G, K])
    sigmas = np.zeros([L, G, K])
    
    resovoir = []
    for l in range(L):
        temp = []
        for k in range(K):
            temp.append([])
        resovoir.append(temp)
    
    statistics, predicates = db.pepsi(K * n1)
    for l in range(L):
        for g in range(G):
            for k in range(K):
                if len(predicates[:, l, g, k]) > 0:
                    ps[l, g, k] = np.mean(predicates[:, l, g, k])
                if np.sum(predicates[:, l, g, k]) > 1:
                    sigmas[l, g, k] = np.std(statistics[predicates[:, l, g, k]], ddof=1)
                resovoir[l][k] = [statistics, predicates[:, l, :, k]]
                    
    t = np.zeros([L, K])
    for l in range(L):
        t[l] = np.sqrt(ps[l, l]) * sigmas[l, l]
        t[l] /= t[l].sum()
                                        
    perfs = np.zeros([L, G])
    for l in range(L):
        for g in range(G):
            t[t == 0] = 1e-8
            temp = ps[l, g] * (sigmas[l, g] ** 2)
            temp /= (ps[l, g].sum() ** 2) * t[l]
            perfs[l, g] = np.sum(temp)
       
    def objective(x):
        weighted_perfs = np.zeros(perfs.shape)
        for l in range(L):
            if x[l] == 0:
                x[l] = 1e-8
            weighted_perfs[l] = perfs[l] * (1 / x[l])
        weighted_perfs = 1 / np.sum(1 / weighted_perfs, axis=0)
        return np.amax(weighted_perfs)

    def constraint(x):
        return np.sum(x) - 1

    @my.penalty.quadratic_inequality(constraint)
    def penalty(x):
        return 0.0
        
    if version == "optimal":
        bounds = [(0, 1)] * L
        guess = np.array([1] * L) / L
        solver = my.solvers.fmin
        mon = my.monitors.Null()
        kwds = dict(disp=False, full_output=True, itermon=mon,
                    args=(),  xtol=1e-8, ftol=1e-8, maxfun=10000, maxiter=10000)
        result = solver(objective, guess, bounds=bounds, penalty=penalty, **kwds)
        n2_weighting = result[0]
        n2_weighting /= n2_weighting.sum()
    else:
        n2_weighting = np.array([1] * L) / L
    
#     print(perfs)
#     print(n2_weighting)
#     print(objective(n2_weighting))
#     weighted_perfs = np.zeros(perfs.shape)
#     for l in range(L):
#         weighted_perfs[l] = perfs[l] * (1 / n2_weighting[l])
#     weighted_perfs = 1 / np.sum(1 / weighted_perfs, axis=0)
#     print("t", weighted_perfs)
        
    ms = np.zeros([L, G, K])
    ps = np.zeros([L, G, K])
                                        
    for l in range(L):
        for k in range(K):
            statistics, predicates = db.sample((n2_weighting[l] * n2 * t[l, k]).astype(int), l, k)
            if len(resovoir[l][k][0]) > 0:
                statistics = np.concatenate([statistics, resovoir[l][k][0]])
            if len(resovoir[l][k][1]) > 0:
                predicates = np.concatenate([predicates, resovoir[l][k][1].T], axis=1)
            for g in range(G):
                if len(predicates[g]) > 0:
                    ps[l, g, k] = np.mean(predicates[g])
                if np.sum(predicates[g]) > 0:
                    ms[l, g, k] = np.mean(statistics[predicates[g]])
                    
    ms = (ms * ps).sum(axis=2) / ps.sum(axis=2)
    weighted_perfs = np.zeros(perfs.shape)
    for l in range(L):
        weighted_perfs[l] = perfs[l] * (1 / n2_weighting[l])
    weighted_perfs = 1 / weighted_perfs
    weighted_perfs = weighted_perfs / weighted_perfs.sum(axis=0) 
    return (ms * weighted_perfs).sum(axis=0)

def motts(db, n1, n2):
    L = db.g
    G = db.g
    K = db.k
    
    n = n1 * K + n2
    
    ms = np.zeros([L, G])
    sigmas = np.zeros([L, G])
    for l in range(L):
        statistics, predicates = db.sample(n // L, l)
        for g in range(G):
            if np.sum(predicates[g]) > 0:
                ms[l, g] = np.mean(statistics[predicates[g]])
            if np.sum(predicates[g]) > 0:
                sigmas[l, g] = np.std(statistics[predicates[g]])
            
    perfs = 1 / (sigmas ** 2)
    perfs = perfs / perfs.sum(axis=0)
    return (ms * perfs).sum(axis=0)



dblgk.ground_truth


foo(dblgk, 10000 // 5, 10000, version="optimal"), motts(dblgk, 10000 // 5, 10000)


@ray.remote
def bar(db, n1, n2, trials=5, version="optimal"):
    estimates = np.zeros([trials, db.g])
    for trial in range(trials):
        estimates[trial] = foo(db, n1, n2, version)
    return estimates

@ray.remote
def gummies(db, n1, n2, trials=5):
    estimates = np.zeros([trials, db.g])
    for trial in range(trials):
        estimates[trial] = motts(db, n1, n2)
    return estimates


C = 1/2
TRIALS = 1000
ns = np.arange(4000, 40500, 4000)
n1s = (ns*C/dblgk.k).astype(np.int32)
n2s = (ns*(1-C)).astype(np.int32)

ours = [None]*len(ns)
normals = [None]*len(ns)
uniforms = [None]*len(ns)

for idx, (n1, n2) in enumerate(zip(n1s, n2s)):
    ours[idx] = bar.remote(dblgk, n1, n2, trials=TRIALS, version="optimal")
    normals[idx] = bar.remote(dblgk, n1, n2, trials=TRIALS, version="normal")
    uniforms[idx] = gummies.remote(dblgk, n1, n2, trials=TRIALS)
    
ours = np.array(ray.get(ours))
normals = np.array(ray.get(normals))
uniforms = np.array(ray.get(uniforms))


error_ours = np.amax(np.mean((ours - dblgk.ground_truth)**2, axis=1), axis=1)
error_normals = np.amax(np.mean((normals - dblgk.ground_truth)**2, axis=1), axis=1)
error_uniforms = np.amax(np.mean((uniforms - dblgk.ground_truth)**2, axis=1), axis=1)


plt.plot(ns, error_ours, label="ours")
plt.plot(ns, error_normals, label="normal")
plt.plot(ns, error_uniforms, label="uniform")
plt.legend()


import pandas as pd

results = {}

for g in range(dblgk.g):
    results[f"ground_truth_{g}"] = [dblgk.ground_truth[g]] * TRIALS
for i in range(len(ns)):
    for g in range(dblgk.g):
        results[f"ours_{ns[i]}_{g}"] = ours[i, :, g]
        results[f"normal_{ns[i]}_{g}"] = normals[i, :, g]
        results[f"uniform_{ns[i]}_{g}"] = uniforms[i, :, g]
        
df = pd.DataFrame.from_dict(results)
df.to_csv("./results/groupby/synthetic-single-oracle.csv")
