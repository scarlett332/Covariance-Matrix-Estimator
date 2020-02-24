import csv
import copy
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from scipy import stats
from estimator import Estimator



def main():
    #Config parameters
    parser = argparse.ArgumentParser(description="Portfolio Optimization")

    parser.add_argument("--market-csv-file", type=str, default="data/S&P500.csv",\
                        help = "location of market information")
    parser.add_argument("--stock-csv-file", type=str, default="data/stock_clean.csv",\
                        help = "location of stock information")
    parser.add_argument("--start-date", type=str, default="19720731",\
                        help = "start date of calculation")
    parser.add_argument("--est-type", type=int, default=0,\
                help = "type of estimator: 0 -> Sample Covariance Estimator, 1-> Market Estimator, 2-> Shrinkage Estimator")
    args = parser.parse_args()

    #Preprocess the data
    df_market = pd.read_csv(args.market_csv_file)
    df_stock = pd.read_csv(args.stock_csv_file)
    df_stock = df_stock.rename(columns = {df_stock.columns[0]:"date"})
    result = df_stock["date"].isin([args.start_date])
    start_month = df_stock["date"][result].index.values[0]

    #Extract the date column
    date = df_stock[df_stock.columns[0]]
    df_stock = df_stock[df_stock.columns[1:]]
    df_market = df_market[df_market.columns[1:]]


    #Initialize
    estimator = Estimator()
    est_types = ["Sample Covariance Estimator",\
                 "Market Estimator",\
                 "Shrinkage Estimator"]
    monthly_returns = np.array([])

    for i in tqdm(range(start_month,384,12)):
        tmp_stock_info = None
        tmp_market_info = None
        #Get in sample data
        tmp_df_stock_all = df_stock[i-119:i+13].dropna(thresh=df_stock[i-119:i+13].shape[0],how='all',axis=1) #Drop columns that have less than pre 120 + post 12 = 132 valid return

        #Extract data
        tmp_stock_info = tmp_df_stock_all[0:120].astype(np.float64).values
        tmp_market_info = df_market[i-119:i+1].astype(np.float64).values
        N = tmp_stock_info.shape[1]
        
        #Calculate corresponding estimators
        cov_matrix = estimator.estimate(est_type = args.est_type,\
                                         stock_info=tmp_stock_info,\
                                         market_info=tmp_market_info)
        #Calculate weight
        cov_matrix_inv = np.linalg.pinv(cov_matrix)
        weight = cov_matrix_inv@np.ones((N,1))/(np.ones((1,N))@cov_matrix_inv@np.ones((N,1)))
 
        #Calculate out of samples weighted returns: in month
        post_returns = tmp_df_stock_all[120:].astype(np.float64).values
        monthly_return = (post_returns@weight).squeeze()
        monthly_returns = np.concatenate((monthly_returns,monthly_return))
    
    #Calculate the annual std by annual.std = monthly.std * sqrt(12)
    annual_std = monthly_returns.std()*np.sqrt(12)
    print("Annual std: {:2f}% estimated by {}".format(100.*annual_std, est_types[args.est_type]))

if __name__ == "__main__":
    main()
