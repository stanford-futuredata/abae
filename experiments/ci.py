import ray
import pandas as pd
import scipy.optimize
import abae
import numpy as np
from tqdm.autonotebook import tqdm

ray.init()
K = 5
C = 1/2
TRIALS = 1000

dbs = [
    abae.JacksonRecords(k=K),
    abae.TaipeiRecords(k=K),
    abae.CelebARecords(k=K),
    abae.MovieFacesV2Records(k=K),
    abae.Trec05PRecords(k=K),
    abae.AmazonOfficeSuppliesRecords(k=K),
]

for db in tqdm(dbs):
    ns = np.arange(500, 10500, 500)
    n1s = (ns*C/db.k).astype(np.int64)
    n2s = (ns*(1-C)).astype(np.int64)

    ours = [None]*len(ns)
    for idx, (n1, n2) in enumerate(zip(n1s, n2s)):
        ours[idx] = abae.execute_ours_with_ci.remote(db, n1, n2, trials=TRIALS) 
    ours, ours_lower_bounds, ours_upper_bounds = zip(*ray.get(ours))
    ours, ours_lower_bounds, ours_upper_bounds = map(np.array, [ours, ours_lower_bounds, ours_upper_bounds])
    ours_widths = ours_upper_bounds - ours_lower_bounds

    uniform = [None]*len(ns)
    for idx, (n1, n2) in enumerate(zip(n1s, n2s)):
        uniform[idx] = abae.execute_uniform_with_ci.remote(db, n1, n2, trials=TRIALS) 
    uniform, uniform_lower_bounds, uniform_upper_bounds = zip(*ray.get(uniform))
    uniform, uniform_lower_bounds, uniform_upper_bounds = map(np.array, [uniform, uniform_lower_bounds, uniform_upper_bounds])
    uniform_widths = uniform_upper_bounds - uniform_lower_bounds

    results = {}
    results[str({"K": K, "C": C, "TRIALS": TRIALS})] = [""]*len(uniform[0])
    results[f"truth"] = [0]*len(uniform[0])
    for i in range(len(ns)):
        results[f"uniform_width_{ns[i]}"] = uniform_widths[i]
    for i in range(len(ns)):
        results[f"ours_width_{ns[i]}"] = ours_widths[i]

    df = pd.DataFrame.from_dict(results)
    df.to_csv(f"./results/ci/{db.name}_ci.csv")
    
ray.shutdown()
