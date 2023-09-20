import pandas as pd
import numpy as np

df = pd.read_csv('./data/jackson.csv')
smooth = 1
df['proxy_scores'] = 1 * smooth + df['proxy_scores'] * (1 - smooth)
df['sqrt'] = np.sqrt(df['proxy_scores'])
df['sqrt'] = df['sqrt'] / np.sum(df['sqrt'])
print(df)

df_pred = df[df['predicates']]
true_val = np.sum(df_pred['statistics']) / len(df_pred)

N = 500
TRIALS = 100
statistics = []
for trial in range(TRIALS):
    inds = np.random.choice(np.arange(len(df)), size=N, p=df['sqrt'])
    df_sample = df.iloc[inds]
    df_rejection = df_sample[df_sample['predicates']]
    statistic = np.sum(df_rejection['sqrt'] * df_rejection['statistics']) / np.sum(df_rejection['sqrt'])
    statistics.append(statistic)

statistics = np.array(statistics)
squared_errors = (statistics - true_val) ** 2
mse = np.average(squared_errors)
print(mse)
print(np.sqrt(mse))
