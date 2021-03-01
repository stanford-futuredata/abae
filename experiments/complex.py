import ray
import pandas as pd
import scipy.optimize
import abae
import numpy as np
from tqdm.autonotebook import tqdm

ray.init()
K = 5
C = 1/2
TRIALS = 2000

dbs = [
    abae.JacksonRedLightMultProxyRecords(k=K),
    abae.JacksonRedLightCarProxyRecords(k=K),
    abae.JacksonRedLightLightProxyRecords(k=K),
    abae.SyntheticComplexPredicatesRecords(k=K, version="opt"),
    abae.SyntheticComplexPredicatesRecords(k=K, version="left"),
    abae.SyntheticComplexPredicatesRecords(k=K, version="right"),
]

for db in tqdm(dbs):
    ns = np.arange(500, 10500, 500)
    n1s = (ns*C/db.k).astype(np.int64)
    n2s = (ns*(1-C)).astype(np.int64)

    ours = [None]*len(ns)
    for idx, (n1, n2) in enumerate(zip(n1s, n2s)):
        ours[idx] = abae.execute_ours.remote(db, n1, n2, trials=TRIALS, sample_reuse=True) 
    ours = np.array(ray.get(ours))

    uniform = [None]*len(ns)
    for idx, (n1, n2) in enumerate(zip(n1s, n2s)):
        uniform[idx] = abae.execute_uniform.remote(db, n1, n2, trials=TRIALS) 
    uniform = np.array(ray.get(uniform))
    
    results = {}
    results[str({"K": K, "C": C, "TRIALS": TRIALS})] = [""]*len(uniform[0])
    results[f"truth"] = [db.ground_truth]*len(uniform[0])
    for i in range(len(ns)):
        results[f"uniform_{ns[i]}"] = uniform[i]
    for i in range(len(ns)):
        results[f"ours_{ns[i]}"] = ours[i]

    df = pd.DataFrame.from_dict(results)
    df.to_csv(f"./results/complex/{db.name}_mse.csv")
    
ray.shutdown()
