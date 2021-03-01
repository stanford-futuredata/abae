import ray
import pandas as pd
import scipy.optimize
import abae
import numpy as np
from tqdm.autonotebook import tqdm

ray.init()
K = 5
C = 1/2
N = 10000
TRIALS = 1000

dbs = [
    abae.JacksonRecords(k=K),
    abae.TaipeiRecords(k=K),
    abae.CelebARecords(k=K),
    abae.MovieFacesV2Records(k=K),
    abae.Trec05PRecords(k=K),
    abae.AmazonOfficeSuppliesRecords(k=K)
]

for db in tqdm(dbs):
    ns = np.arange(500, 10500, 500)
    n1s = (ns*C/db.k).astype(np.int64)
    n2s = (ns*(1-C)).astype(np.int64)

    ours = [None]*len(ns)
    for idx, (n1, n2) in enumerate(zip(n1s, n2s)):
        ours[idx] = abae.execute_ours.remote(db, n1, n2, trials=TRIALS, sample_reuse=True) 
    ours = np.array(ray.get(ours))

    ours2 = [None]*len(ns)
    for idx, (n1, n2) in enumerate(zip(n1s, n2s)):
        ours2[idx] = abae.execute_ours.remote(db, n1, n2, trials=TRIALS, sample_reuse=False) 
    ours2 = np.array(ray.get(ours2))

    uniforms = [None]*len(ns)
    for idx, (n1, n2) in enumerate(zip(n1s, n2s)):
        uniforms[idx] = abae.execute_uniform.remote(db, n1, n2, trials=TRIALS) 
    uniforms = np.array(ray.get(uniforms))
    
    results = {}
    results[str({"K": K, "C": C, "TRIALS": TRIALS})] = [""]*len(uniforms[0])
    results[f"truth"] = [db.ground_truth]*len(uniforms[0])
    for i in range(len(ns)):
        results[f"uniform_{ns[i]}"] = uniforms[i]
    for i in range(len(ns)):
        results[f"ours_sample_reuse_{ns[i]}"] = ours[i]
    for i in range(len(ns)):
        results[f"ours_no_sample_reuse_{ns[i]}"] = ours2[i]

    df = pd.DataFrame.from_dict(results)
    df.to_csv(f"./results/lesion/{db.name}_sample_reuse.csv")
    
ray.shutdown()
