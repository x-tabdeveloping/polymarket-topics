# maybe clip logit transform?

# see https://stats.stackexchange.com/questions/161639/does-the-dickey-fuller-test-for-a-random-walk for ADF. If p-val > .05 -> unit root -> random walk


import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller
from scipy.special import logit


from hurst import compute_Hc

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

    #make list if market id is in filtered and non filtered dfs
    include_idx = df["id"].isin(new_df["id"])

    #assign to input df
    df["include"] = include_idx

    #return input df
    return df

#def augmented dickey-fuller test
def adf_test(ts):

    test = adfuller(ts, regression = "c")
    dfoutput = pd.Series(
    test[0:4],
    index=[
        "Test Statistic",
        "p-value",
        "#Lags Used",
        "Number of Observations Used",
    ],
)
    for key, value in test[4].items():
        dfoutput["Critical Value (%s)" % key] = value

    return dfoutput


#make absolute error function
def abs_error(p, token_correct):

    abs_error = np.abs(p-token_correct)

    return abs_error


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

if __name__ == "__main__":
# read in data
    prices = pd.read_csv("data/prices.csv")
    meta_data = pd.read_csv("data/market_metadata.csv")

    #make datetime col
    prices["datetime"] = pd.to_datetime(prices["datetime"])

    # reconstruct outcomes
    prices = prep_price_data(prices, meta_data)

    #calc brier for each price
    prices = prices.assign(brier = calc_brier(p = prices["p"], token_correct=prices["token_correct"]))
    prices = prices.assign(abs_error = abs_error(p = prices["p"], token_correct=prices["token_correct"]))

    #filter for outcome _index 1
    prices = prices[prices["outcome_index"] == 1]
    
    records = []
    for market_id in tqdm(prices["market_id"].unique(), desc="Calculating features for all markets."):

        #get market
        market = prices[prices["market_id"]==market_id]

        #get include value
        include = market["include"].iloc[0]

        #get mean scores
        mean_brier = market["brier"].mean()
        mean_abs_error = market["abs_error"].mean()

        #sort series
        market = market.sort_values("datetime")

        #move series to log_odds
        ts = market["p"].clip(1e-6, 1 - 1e-6).map(logit)

        #make t-t_1 diff
        residuals = (ts - ts.shift(1)).dropna()

        #get drift and sd
        drift = residuals.mean()
        vol = residuals.std()

        #try adf
        try:
            adf_res = adf_test(ts)
            adf_test_stat = adf_res["Test Statistic"]
            adf_p_val = adf_res["p-value"]
        except Exception:
            adf_test_stat = np.nan
            adf_p_val = np.nan

        #try hurst
        try:
            H, c, data = compute_Hc(ts, "random_walk")
        except Exception:
            H, c = np.nan, np.nan

        records.append(dict(
            market_id=market_id,
            include=include,
            mean_brier=mean_brier,
            mean_abs_error=mean_abs_error,
            drift=drift,
            vol=vol,
            adf_test_stat=adf_test_stat,
            adf_p_val=adf_p_val,
            hurst_exponent=H,
            hurst_c=c,
        ))



    results = pd.DataFrame.from_records(records)
    results.to_csv("data/features.csv", index = False)

    



