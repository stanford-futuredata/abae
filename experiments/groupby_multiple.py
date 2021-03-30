import ray
import pandas as pd
import scipy.optimize
import abae
import numpy as np
from tqdm.autonotebook import tqdm
from tabulate import tabulate
import matplotlib.pyplot as plt
ray.init(ignore_reinit_error=True)
K = 5
C = 1/2
TRIALS = 1000

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


db_gray = abae.Records(5,
    gray["proxy_scores"][gray_sort][:, 0],
    smiling_statistics,
    np.array(gray["predicates"])[gray_sort],
)
print("GRAY")
db_gray.summary()

db_blond = abae.Records(5,
    blond["proxy_scores"][blond_sort][:, 0],
    smiling_statistics,
    np.array(blond["predicates"])[blond_sort],
)
print("\nBLOND")
db_blond.summary()

db_black = abae.Records(5,
    black["proxy_scores"][black_sort][:, 0],
    smiling_statistics,
    np.array(black["predicates"])[black_sort],
)
print("\nBLACK")
db_black.summary()

db_brown = abae.Records(5,
    brown["proxy_scores"][brown_sort][:, 0],
    smiling_statistics,
    np.array(brown["predicates"])[brown_sort],
)
print("\nBROWN")
db_brown.summary()


TRIALS = 1000
dbs = [db_gray, db_blond]
C = 1/2
ns = np.arange(4000, 20500, 1000)
n1s = (ns*C/dbs[0].k/len(dbs)).astype(np.int64)
n2s = (ns*(1-C)).astype(np.int64)

ours1 = [None]*len(ns)
ours2 = [None]*len(ns)
ours3 = [None]*len(ns)
for idx, (n1, n2) in enumerate(zip(n1s, n2s)):
    ours1[idx] = abae.execute_group_by_ours_v2.remote(dbs, n1, n2, trials=TRIALS, version="optimal")
    ours2[idx] = abae.execute_group_by_ours_v2.remote(dbs, n1, n2, trials=TRIALS, version="troll")
    ours3[idx] = abae.execute_group_by_uniform_v2.remote(dbs, n1, n2, trials=TRIALS)
ours1 = np.array(ray.get(ours1))
ours2 = np.array(ray.get(ours2))
ours3 = np.array(ray.get(ours3))

errors1 = np.amax(np.mean((ours1 - [db.ground_truth for db in dbs])**2, axis=1), axis=1)
errors2 = np.amax(np.mean((ours2 - [db.ground_truth for db in dbs])**2, axis=1), axis=1)
errors3 = np.amax(np.mean((ours3 - [db.ground_truth for db in dbs])**2, axis=1), axis=1)


plt.plot(ns, errors1, color="b", label="optimized")
plt.plot(ns, errors2, color="g", label="equal n algx")
plt.plot(ns, errors3, color="r", label="uniform")
plt.legend()


results = {}
for i, db in enumerate(dbs):
    results[f"ground_truth_{i}"] = [db.ground_truth] * TRIALS
    
for i in range(len(ns)):
    for j in range(len(dbs)):
        results[f"optimal_{ns[i]}_{j}"] = ours1[i, :, j]
        results[f"equal_n2_{ns[i]}_{j}"] = ours2[i, :, j]
        results[f"uniform_{ns[i]}_{j}"] = ours3[i, :, j]
        
df = pd.DataFrame.from_dict(results)
df.to_csv(f"/future/u/jtguibas/aggpred/results/groupby/celeba-multiple-oracle.csv")
df


N = 1000000
G = 4
K = 5

ps = np.array([
    [0.3, 0.3, 0.5, 1, 1],
    [0.1, 0.3, 0.3, 0.5, 1], 
    [0.1, 0.1, 0.3, 0.3, 0.5], 
    [0.1, 0.1, 0.1, 0.3, 0.3], 
])

ps /= 5

proxy_scores = np.zeros([G, N])
statistics = np.random.normal(5, 5, N)
predicates = np.zeros([G, N]).astype(bool)

for i in range(N):
    k = i // (N // K)
    proxy_scores[:, i] = ps[:, k]
    if ps[:, k].sum() >= 1 or np.random.binomial(1, ps[:, k].sum()) == 1:
        dist = ps[:, k] / ps[:, k].sum()
        g = np.random.choice(np.arange(0, G), p=dist)
        predicates[g, i] = True
    
db1 = abae.Records(K, proxy_scores[0], statistics, predicates[0])
db2 = abae.Records(K, proxy_scores[1], statistics, predicates[1])
db3 = abae.Records(K, proxy_scores[2], statistics, predicates[2])
db4 = abae.Records(K, proxy_scores[3], statistics, predicates[3])


db1.summary()
db2.summary()
db3.summary()
db4.summary()


TRIALS = 100
dbs = [db1, db2, db3, db4]
C = 1/2
ns = np.arange(8000, 40500, 4000)
n1s = (ns*C/dbs[0].k/len(dbs)).astype(np.int64)
n2s = (ns*(1-C)).astype(np.int64)

ours1 = [None]*len(ns)
ours2 = [None]*len(ns)
ours3 = [None]*len(ns)
for idx, (n1, n2) in enumerate(zip(n1s, n2s)):
    ours1[idx] = abae.execute_group_by_ours_v2.remote(dbs, n1, n2, trials=TRIALS, version="optimal")
    ours2[idx] = abae.execute_group_by_ours_v2.remote(dbs, n1, n2, trials=TRIALS, version="troll")
    ours3[idx] = abae.execute_group_by_uniform_v2.remote(dbs, n1, n2, trials=TRIALS)
ours1 = np.array(ray.get(ours1))
ours2 = np.array(ray.get(ours2))
ours3 = np.array(ray.get(ours3))

errors1 = np.amax(np.mean((ours1 - [db.ground_truth for db in dbs])**2, axis=1), axis=1)
errors2 = np.amax(np.mean((ours2 - [db.ground_truth for db in dbs])**2, axis=1), axis=1)
errors3 = np.amax(np.mean((ours3 - [db.ground_truth for db in dbs])**2, axis=1), axis=1)


plt.plot(ns, errors1, color="b", label="optimized")
plt.plot(ns, errors2, color="g", label="equal n algx")
plt.plot(ns, errors3, color="r", label="uniform")
plt.legend()


results = {}
for i, db in enumerate(dbs):
    results[f"ground_truth_{i}"] = [db.ground_truth] * TRIALS
    
for i in range(len(ns)):
    for j in range(len(dbs)):
        results[f"optimal_{ns[i]}_{j}"] = ours1[i, :, j]
        results[f"equal_n2_{ns[i]}_{j}"] = ours2[i, :, j]
        results[f"uniform_{ns[i]}_{j}"] = ours3[i, :, j]
        
df = pd.DataFrame.from_dict(results)
df.to_csv(f"/future/u/jtguibas/aggpred/results/groupby/synthetic-multiple-oracle.csv")
