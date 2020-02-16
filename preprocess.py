import csv
import copy
from tqdm import tqdm
import pandas as pd
import numpy as np

def main():
    csv_file = "data/stock.csv"
    out_file = "data/stock_clean.csv"
    companys = {}
    #companys_copy = {}
    with open('data/stock.csv', newline='') as f:
        reader = csv.reader(f) # Code, Year, Company, Exchange, Return
        for idx,row in enumerate(reader):
            if idx==0:
                continue
            if not row[2] in companys:
                companys[row[2]] = {}
            else:
                companys[row[2]][row[1]] = row[-1]
    #To dataframe
    df = pd.DataFrame(companys)
    
    #Sort date
    df.sort_index(axis=0,inplace = True)
    
    #Delete columns with less 120 months available
    cols = df.columns
    for col in tqdm(cols):
        if np.sum(df[col].notna())<=120:
            df.drop(col,axis=1,inplace = True)
    
    #Save to csv
    df.to_csv(out_file)
if __name__ == "__main__":
    main()
