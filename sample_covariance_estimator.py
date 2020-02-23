import csv
import copy
from tqdm import tqdm
import pandas as pd
import numpy as np

csv_file = "data/stock_clean.csv"
df = pd.read_csv(csv_file)

df = df.rename(columns = {df.columns[0]:"date"})
result = df["date"].isin(["19720731"])
t = df["date"][result].index.values[0]+1
print(t)


monthly_returns = []
for i in tqdm(range(t,384,12)):
    #Drop columns that have less than pre 120 + post 12 = 132 valid return
    tmp_df_all = df[i-119:i+13].dropna(thresh=df[i-119:i+13].shape[0],how='all',axis=1)
    #Get the prev
    tmp_df = tmp_df_all[0:120]
    N = len(tmp_df.columns[1:])
    cov_matrix = tmp_df[tmp_df.columns[1:]].astype(np.float64).cov()
    cov_matrix_inv = pd.DataFrame(np.linalg.pinv(cov_matrix.values), cov_matrix.columns, cov_matrix.index)
    mu = tmp_df[tmp_df.columns[1:]].mean()
    q = 0.2*tmp_df[tmp_df.columns[1:]].astype(np.float64).mean().mean()
    A = np.ones((1,N))@cov_matrix_inv.values@np.ones((N,1))
    B = np.ones((1,N))@cov_matrix_inv.values@mu.values
    C = mu.values.T@cov_matrix_inv.values@mu.values
    #Calculate the weight
    w = cov_matrix_inv.values@np.ones((N,1))/(np.ones((1,N))@cov_matrix_inv.values@np.ones((N,1)))

    #Calculate returns from outsamples
    for j in range(12):
        tmp_df_post = tmp_df_all[120+j:121+j]
        tmp_df_post = tmp_df_post[tmp_df_post.columns[1:]]
        annual_return = (tmp_df_post.astype(np.float64)+1).prod(axis=0)-1
        monthly_returns.append(annual_return@w)

monthly_returns = np.array(monthly_returns)
print("Annual std: {:2f}%".format(100.*monthly_returns.std()*np.sqrt(12)))
