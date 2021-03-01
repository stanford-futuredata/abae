import pandas as pd
import numpy as np
from tabulate import tabulate

HOME = "/future/u/jtguibas/abae/data/"

class Records:
    def __init__(self, k, proxy_scores, statistics, predicates):
        assert(proxy_scores.shape == predicates.shape == statistics.shape)
        self.k = k
        self.proxy_scores = proxy_scores
        self.statistics = statistics
        self.predicates = predicates

        self.sort = np.argsort(proxy_scores)
        self.statistics_sorted = statistics[self.sort]
        self.predicates_sorted = predicates[self.sort]
        self.ground_truth = statistics[predicates].mean()
        
        self.p = np.sum(self.predicates) / len(self.proxy_scores)
        self.sigma = np.std(self.statistics[self.predicates])
        self.strata = np.array_split(np.arange(self.sort.shape[0]), self.k)
        self.ps = np.zeros(self.k)
        self.sigmas = np.zeros(self.k)
        self.ms = np.zeros(self.k)
        for i in range(self.k):
            stratum = self.strata[i]
            _statistics = self.statistics_sorted[stratum].copy()
            _predicates = self.predicates_sorted[stratum].copy()
            self.ps[i] = np.sum(_predicates) / len(_predicates)
            self.sigmas[i] = np.std(_statistics[_predicates])
            self.ms[i] = np.mean(_statistics[_predicates])

    def sample(self, n, k=None):
        if k is None:
            sample_idxs = np.random.choice(self.sort.shape[0], n, replace=False)
        else:
            strata = np.array_split(np.arange(self.sort.shape[0]), self.k)[k]
            sample_idxs = np.random.choice(strata, n, replace=False)

        statistics = self.statistics_sorted[sample_idxs].copy()
        predicates = self.predicates_sorted[sample_idxs].copy()
        return statistics, predicates
    
    def summary(self):
        num_records = len(self.proxy_scores)
        table = tabulate(
            [
                ["NUM_RECORDS", num_records],
                ["K", self.k],
                ["P_S", np.round(self.ps.sum() / self.k, 5)],
                ["SIGMA_S", np.round(self.sigma, 5)],
                ["P_K", np.round(self.ps, 5)],
                ["SIGMA_K", np.round(self.sigmas, 5)],
                ["M_K", np.round(self.ms, 5)],
            ],
            headers=["Key", "Value"]
        )
        print(table)
        
    
class JacksonRecords(Records):
    def __init__(self, k):
        self.name = "jackson"
        df = pd.read_csv(HOME + f"{self.name}.csv")
        proxy_scores = df["proxy_scores"].to_numpy()
        statistics = df["statistics"].to_numpy()
        predicates = df["predicates"].to_numpy()
        super().__init__(k, proxy_scores, statistics, predicates)
        
        
class JacksonRedLightRecords(Records):
    def __init__(self, k):
        self.name = "jackson_red_light"
        df = pd.read_csv(HOME + f"{self.name}.csv")
        proxy_scores = df["proxy_scores"].to_numpy()
        statistics = df["statistics"].to_numpy()
        predicates = df["predicates"].to_numpy()
        super().__init__(k, proxy_scores, statistics, predicates)
        
        
class AmazonOfficeSuppliesRecords(Records):
    def __init__(self, k):
        self.name = "amazon_office"
        df = pd.read_csv(HOME + f"{self.name}.csv")
        proxy_scores = df["proxy_scores"].to_numpy()
        statistics = df["statistics"].to_numpy()
        predicates = df["predicates"].to_numpy()
        super().__init__(k, proxy_scores, statistics, predicates)
        
        
class MovieFacesV2Records(Records):
    def __init__(self, k):
        self.name = "moviefacesv2"
        proxy_scores = np.load("/future/u/jtguibas/aggpred/data/movie-faces-proxy-score-v3.npy")[:, 0]
        predicates = np.load("/future/u/jtguibas/aggpred/data/movie-faces-predicates-v2.npy")
        statistics = np.load("/future/u/jtguibas/aggpred/data/movie-faces-statistics-v2.npy")
        super().__init__(k, proxy_scores, statistics, predicates)
        
        
class CelebARecords(Records):
    def __init__(self, k):
        self.name = "celeba"
        df = pd.read_csv(HOME + f"{self.name}.csv")
        proxy_scores = df["proxy_scores"].to_numpy()
        statistics = df["statistics"].to_numpy()
        predicates = df["predicates"].to_numpy()
        super().__init__(k, proxy_scores, statistics, predicates)    
        

class Trec05PRecords(Records):
    def __init__(self, k):
        self.name = "trec05p"
        df = pd.read_csv(HOME + f"{self.name}.csv")
        proxy_scores = df["proxy_scores"].to_numpy()
        statistics = df["statistics"].to_numpy()
        predicates = df["predicates"].to_numpy()
        super().__init__(k, proxy_scores, statistics, predicates)
        
class TaipeiRecords(Records):
    def __init__(self, k):
        self.name = "taipei"
        df = pd.read_csv(HOME + f"{self.name}.csv")
        proxy_scores = df["proxy_scores"].to_numpy()
        statistics = df["statistics"].to_numpy()
        predicates = df["predicates"].to_numpy()
        super().__init__(k, proxy_scores, statistics, predicates) 
        
        
class SyntheticRecords(Records):
    def __init__(self, k, alpha=0.1, beta=0.5, N=1000000):
        self.name = "synthetic"
        rng = np.random.RandomState(3212142)
        proxy_scores = rng.beta(alpha, beta, size=N)
        statistics = rng.normal(10, 3, N)
        predicates = rng.binomial(n=1, p=proxy_scores).astype(bool)
        super().__init__(k, proxy_scores, statistics, predicates)
        
        
class SyntheticControlRecords(Records):
    def __init__(self, k, ps, sigmas, ms, N=1000000):
        self.name = "synthetic_control"
        rng = np.random.RandomState(3212142)
        strata_size = N // k
        proxy_scores = []
        statistics = np.concatenate([rng.normal(ms[i], sigmas[i], N // k) for i in range(k)])
        predicates = []
        for i in range(k):
            a = rng.binomial(n=1, p=[ps[i]]*strata_size)
            c = rng.binomial(n=strata_size, p=a).astype(bool)
            proxy_scores.append(a)
            predicates.append(c)
        proxy_scores = np.arange(N)
        predicates = np.concatenate(predicates)
        super().__init__(k, proxy_scores, statistics, predicates)
        
        
class SyntheticComplexPredicatesRecords(Records):
    def __init__(self, k, version="opt"):
        self.name = f"synthetic_complex_predicates_{version}"
        N = 1000000
        rng = np.random.RandomState(3212142)

        proxy_scores_a = rng.beta(0.4, 1, N)
        proxy_scores_b = rng.beta(0.2, 1, N)

        proxy_scores_gt = proxy_scores_a * proxy_scores_b

        if version == "opt":
            proxy_scores = proxy_scores_gt.copy()
        elif version == "left":
            proxy_scores = proxy_scores_a
        elif version == "right":
            proxy_scores = proxy_scores_b
        else:
            raise NotImplementedError

        statistics = rng.normal(10, 3, N)
        predicates = np.zeros(N)
        predicates = rng.binomial(n=1, p=proxy_scores_gt)

        predicates = predicates.astype(bool)
        super().__init__(k, proxy_scores, statistics, predicates)
        
        
class JacksonRedLightMultProxyRecords(Records):
    def __init__(self, k):
        self.name = "jackson_red_light_mult"
        df = pd.read_csv(HOME + f"{self.name}.csv")
        proxy_scores = df["proxy_scores"].to_numpy()
        statistics = df["statistics"].to_numpy()
        predicates = df["predicates"].to_numpy()
        super().__init__(k, proxy_scores, statistics, predicates)
        
        
class JacksonRedLightCarProxyRecords(Records):
    def __init__(self, k):
        self.name = "jackson_red_light_car_proxy"
        df = pd.read_csv(HOME + f"{self.name}.csv")
        proxy_scores = df["proxy_scores"].to_numpy()
        statistics = df["statistics"].to_numpy()
        predicates = df["predicates"].to_numpy()
        super().__init__(k, proxy_scores, statistics, predicates)
        
        
class JacksonRedLightLightProxyRecords(Records):
    def __init__(self, k):
        self.name = "jackson_red_light_light_proxy"
        df = pd.read_csv(HOME + f"{self.name}.csv")
        proxy_scores = df["proxy_scores"].to_numpy()
        statistics = df["statistics"].to_numpy()
        predicates = df["predicates"].to_numpy()
        super().__init__(k, proxy_scores, statistics, predicates)