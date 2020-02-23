import csv
import copy
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy import stats


market_csv_file = "data/S&P500.csv"
stock_csv_file = "data/stock_clean.csv"
df_market = pd.read_csv(market_csv_file)
df_stock = pd.read_csv(stock_csv_file)


df_stock = df_stock.rename(columns = {df_stock.columns[0]:"date"})
result = df_stock["date"].isin(["19720731"])
t = df_stock["date"][result].index.values[0]


monthly_returns = []
#range(t,384,12)
for i in tqdm(range(t,384,12)):

    #Drop columns that have less than pre 120 + post 12 = 132 valid return
    tmp_df_stock_all = df_stock[i-119:i+13].dropna(thresh=df_stock[i-119:i+13].shape[0],how='all',axis=1)

    #In sample 
    tmp_df_stock = tmp_df_stock_all[0:120]
    N = len(tmp_df_stock.columns[1:])
    tmp_df_market = df_market[t-119:t+1]
    y = tmp_df_stock[tmp_df_stock.columns[1:]].astype(np.float64).values
    y_bar = y.mean(axis=0)
    x = tmp_df_market[tmp_df_market.columns[1:]].astype(np.float64).values
    x_bar = x.mean()
    beta_1 = np.sum((y-y_bar)*(x),axis=0)/np.sum((x-x_bar)*x,axis=0)
    beta_0 = y_bar - beta_1*x_bar
    residual = y - (beta_1*x+beta_0)
    delta = np.sum(residual**2,axis=0)/(N-2)
    D = np.diag(delta)
    s_2 = x.var()
    F = s_2*beta_1[:,np.newaxis]@beta_1[:,np.newaxis].T + D
    S = tmp_df_stock[tmp_df_stock.columns[1:]].astype(np.float64).cov()
    
    
    pi = 0
    rou = 0
    gamma = 0
    T = 120

    yx = np.concatenate((y,x),axis=1)
    yx_cov = np.cov(yx,rowvar=False)
    
    gamma = ((F-S)**2).sum().sum()
    s_x0 = np.expand_dims(yx_cov[-1,:-1],axis=1)
    PI = ((((y-y_bar)**2).T@((y-y_bar)**2)- 2*(y-y_bar).T@(y-y_bar)*S + T*S*S)/T).values
    pi = PI.sum().sum()
    r = np.zeros((N,N))
    s_00 = yx_cov[-1,-1]
    for j in range(T):
        y_e = np.expand_dims(y[j,:]-y_bar,axis=1)
        r+=((x[j]-x_bar)*(s_00*s_x0@y_e.T+s_00*y_e@s_x0.T-s_x0@s_x0.T*(x[j]-x_bar))*(y_e@y_e.T)/(s_00**2) - F*S)
    np.fill_diagonal(r.values,np.diag(PI))
    rou =r.values.sum().sum()
    #print(rou)
    k = (pi-rou)/gamma
    #print(k/T)
    E = k/T * F +(T-k)/T* S
    #print(E)
    
    cov_matrix = E
    cov_matrix_inv = np.linalg.pinv(cov_matrix)
    
    w = cov_matrix_inv@np.ones((N,1))/(np.ones((1,N))@cov_matrix_inv@np.ones((N,1)))
    mu = tmp_df_stock[tmp_df_stock.columns[1:]].mean()
    #print(mu)
    for j in range(12):
        tmp_df_post = tmp_df_stock_all[120+j:121+j]
        tmp_df_post = tmp_df_post[tmp_df_post.columns[1:]]
        monthly_return = (tmp_df_post.astype(np.float64)+1).prod(axis=0)-1
        monthly_returns.append(monthly_return@w)
monthly_returns = np.array(monthly_returns)
print("Annual std: {:2f}%".format(100.*monthly_returns.std()*np.sqrt(12)))