import csv
import copy
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy import stats


market_csv_file = "data/S&P500.csv"
stock_csv_file = "data/stock_clean.csv"
norisk_csv_file = "data/norisk.csv"
df_market = pd.read_csv(market_csv_file)
df_stock = pd.read_csv(stock_csv_file)
df_risk = pd.read_csv(norisk_csv_file)

df_stock = df_stock.rename(columns = {df_stock.columns[0]:"date"})
result = df_stock["date"].isin(["19720731"])
t = df_stock["date"][result].index.values[0]

norisk_return = df_risk["ave_1"]/100

monthly_returns = []
for i in tqdm(range(t,384,12)):
    #Drop columns that have less than pre 120 + post 12 = 132 valid return
    tmp_df_stock_all = df_stock[i-119:i+13].dropna(thresh=df_stock[i-119:i+13].shape[0],how='all',axis=1)
    #Get the prev
    tmp_df_stock = tmp_df_stock_all[0:120]
    N = len(tmp_df_stock.columns[1:])
    tmp_df_market = df_market[t-119:t+1]
    tmp_no_risk = norisk_return[t-119:t+1]
    
    y = tmp_df_stock[tmp_df_stock.columns[1:]].astype(np.float64).values
    y_bar = y.mean()
    x = tmp_df_market[tmp_df_market.columns[1:]].values
    x_bar = x.mean()
    beta_1 = np.sum((y-y_bar)*(x),axis=0)/np.sum((x-x_bar)*x,axis=0)
    beta_0 = y_bar - beta_1*x_bar
    residual = y - (beta_1*x+beta_0)
    delta = np.sum(residual**2,axis=0)/(N-2)
    D = np.diag(delta)
    s_2 = x.var()
    cov_matrix = s_2*beta_1[:,np.newaxis]@beta_1[:,np.newaxis].T + D
    #print((beta_1[:,np.newaxis]@beta_1[:,np.newaxis].T).shape)
    cov_matrix_inv = np.linalg.pinv(cov_matrix)#pd.DataFrame(np.linalg.pinv(cov_matrix.values), cov_matrix.columns, cov_matrix.index)
    w = cov_matrix_inv@np.ones((N,1))/(np.ones((1,N))@cov_matrix_inv@np.ones((N,1)))
    mu = tmp_df_stock[tmp_df_stock.columns[1:]].mean()
    for j in range(12):
        tmp_df_post = tmp_df_stock_all[120+j:121+j]
        tmp_df_post = tmp_df_post[tmp_df_post.columns[1:]]
        #print(tmp_df_post)
        monthly_return = (tmp_df_post.astype(np.float64)+1).prod(axis=0)-1
        monthly_returns.append(monthly_return@w)
monthly_returns = np.array(monthly_returns)
print("Annual std: {:2f}%".format(100.*monthly_returns.std()*np.sqrt(12)))
