import ray
import pandas as pd
import scipy.optimize
import abae
import numpy as np
from tqdm.autonotebook import tqdm

ray.init()
K = 5
C = 1/2
N = 5000
TRIALS = 1000

dbs = [
    abae.JacksonRecords,
    abae.TaipeiRecords,
    abae.CelebARecords,
    abae.MovieFacesV2Records,
    abae.Trec05PRecords,
    abae.AmazonOfficeSuppliesRecords
]

for dbx in tqdm(dbs):
    ks = np.arange(2, 20, 1)
    ours = [None]*len(ks)
    uniforms = [None]*len(ks)
    for idx, k in enumerate(ks):
        db = dbx(k)
        n1 = int(N*C/db.k)
        n2 = int(N*(1-C))
        ours[idx] = abae.execute_ours.remote(db, n1, n2, trials=TRIALS) 
        uniforms[idx] = abae.execute_uniform.remote(db, n1, n2, trials=TRIALS) 
    ours = np.array(ray.get(ours))
    uniforms = np.array(ray.get(uniforms))
    
    results = {}
    results[str({"K": K, "C": C, "TRIALS": TRIALS})] = [""]*len(uniforms[0])
    results[f"truth"] = [db.ground_truth]*len(uniforms[0])
    for i in range(len(ks)):
        results[f"uniform_{ks[i]}"] = uniforms[i]
    for i in range(len(ks)):
        results[f"ours_{ks[i]}"] = ours[i]

    df = pd.DataFrame.from_dict(results)
    df.to_csv(f"./results/sensitivity/{db.name}_k.csv")
    
    
for dbx in tqdm(dbs):
    db = dbx(K)
    cs = np.arange(0.1, 1.0, 0.1)
    
    ours = [None]*len(cs)
    uniforms = [None]*len(cs)
    for idx, c in enumerate(cs):
        n1 = int(N*c/db.k)
        n2 = int(N*(1-c))
        ours[idx] = abae.execute_ours.remote(db, n1, n2, trials=TRIALS) 
        uniforms[idx] = abae.execute_uniform.remote(db, n1, n2, trials=TRIALS) 
    ours = np.array(ray.get(ours))
    uniforms = np.array(ray.get(uniforms))
    
    results = {}
    results[str({"K": K, "C": C, "TRIALS": TRIALS})] = [""]*len(uniforms[0])
    results[f"truth"] = [db.ground_truth]*len(uniforms[0])
    for i in range(len(cs)):
        results[f"uniform_{round(cs[i], 2)}"] = uniforms[i]
    for i in range(len(cs)):
        results[f"ours_{round(cs[i], 2)}"] = ours[i]

    df = pd.DataFrame.from_dict(results)
    df.to_csv(f"./results/sensitivity/{db.name}_c.csv")
    
ray.shutdown()
