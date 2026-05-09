#calculate_brier.py

## Update to account for multiple outcomes

#imports
import pandas as pd
import numpy as np

# get assumed outcome of market based on price of last token
def get_outcome(last_price_list, tolerance=0.01):
    
    #loop over prices
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
            outcomes.append(np.nan)

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

#make squared error given price and token correct indicator
def calc_brier(p, token_correct):

    brier = (p-token_correct)**2

    return brier

# make a function to prep price data. 
# Returns price data with outcome and include columns constructed from the meta data.
def prep_price_data(price_df, meta_data_df):

    #make outcome column
    meta_data_df["resolution"] = get_outcome(meta_data_df.lastTradePrice, tolerance=0.015)

    #make include column
    meta_data_df = make_include_column(meta_data_df)

    # select include, outcome cols. add to prices
    meta_data_df = meta_data_df[["id", "include", "resolution"]]
    price_df = pd.merge(price_df, meta_data_df, left_on = "market_id", right_on="id")

    #make 0-1 indicator column if the token_index matches outcome, ie token was correct. if nan, keep nan
    price_df["token_correct"] = np.where(
        price_df["resolution"].isna(), np.nan,
        (price_df["outcome_index"] == price_df["resolution"]).astype(int)
    )
    
    return price_df


# run script
if __name__ == "__main__":
    
    print("making brier scores")

    #read in meta data
    meta_data = pd.read_csv("data/market_metadata.csv")

    #read prices
    prices = pd.read_csv("data/prices.csv")

    #add include and outcome column to prices
    prices = prep_price_data(price_df=prices, meta_data_df=meta_data)

    #calc brier for each price
    prices = prices.assign(brier = calc_brier(p = prices["p"], token_correct=prices["token_correct"]))
    
    ## get mean for each market
    market_brier_scores = prices.groupby(['id', 'outcome_index', 'include']).agg({'brier': 'mean'})

    ## output to .csv
    market_brier_scores.to_csv("data/brier_scores.csv")
    print("brier scores calculated")
    


