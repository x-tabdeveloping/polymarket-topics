#calculate_brier.py

#imports
import pandas as pd
import numpy as np


def get_outcome(last_price_list, tolerance=0.01):
    
    outcomes = []
    for i, price in enumerate(last_price_list):

        #if more than 1-tolerance percent set to outcome 0
        if price > 1-tolerance:
            outcomes.append(0)

        #else set to opposite outcome, ie set outcome to 1
        elif price < 0+tolerance:
            outcomes.append(1)

        #if price was outside tolerance, set to NA
        else:
            outcomes.append(pd.NA)

    #get number of nas produced.
    num_nas = np.sum(pd.isna(outcomes))

    print(f"Resolutions reconstructed with tolerance: {tolerance}. Implies outcome 0 if last price of 0 is less than {1 - tolerance}, outcome 1 if last price more than {0+tolerance}. \n{num_nas} prices were outside of tolerance.")

    return outcomes

#function for which markets to include in further analysis. 
def make_include_column(df):

    #filter for resolved markets
    new_df = df.loc[df["umaResolutionStatus"] == "resolved"]
    print(f"num markets after filtering for resolved {len(new_df)}")

    #filter for markets where final price was inside tolerance
    new_df = new_df.loc[df["resolution"].isna() == False]
    print(f"num markets after filtering outcomes outside tolerance {len(new_df)}")

    #make list if market id is in filtered and non filtered dfs
    include_idx = df["id"].isin(new_df["id"])

    #assign to input df
    df["include"] = include_idx

    #return input df
    return df

#squared error
def calc_brier(p, outcome):

    brier = (p-outcome)**2

    return brier

if __name__ == "__main__":
    #read in meta data
    meta_data = pd.read_csv("data/market_metadata.csv")

    ## get last trade price
    meta_data["resolution"] = get_outcome(meta_data.lastTradePrice, tolerance=0.015)

    #make include column
    meta_data = make_include_column(meta_data)

    #read prices
    price_data = pd.read_csv("data/prices.csv")

    # select include, outcome cols. add to prices
    meta_data = meta_data[["id", "include", "resolution"]]
    price_data = pd.merge(price_data, meta_data, left_on = "market_id", right_on="id")

    #calc brier for each price
    price_data = price_data.assign(brier = calc_brier(p = price_data["p"], outcome=price_data["resolution"]))
    
    ## get mean for each market
    market_brier_scores = price_data.groupby(['id', 'include']).agg({'brier': 'mean'})

    ## output to .csv
    market_brier_scores.to_csv("data/brier_scores.csv")
    print("brier scores calculated")
    
