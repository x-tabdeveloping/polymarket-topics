import numpy as np
import pandas as pd
from hurst import compute_Hc
from scipy.special import logit
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm


def get_outcome(last_price_list, tolerance=0.01):
    '''Use the value of the last trade price to infer outcome token'''
    outcomes = []
    for i, price in enumerate(last_price_list):
        if price > 1-tolerance:
            outcomes.append(0)
        elif price < 0+tolerance:
            outcomes.append(1)
        else:
            outcomes.append(np.nan)
    num_nas = np.sum(pd.isna(outcomes))
    print(f"Resolutions reconstructed with tolerance: {tolerance}. Implies outcome 0 if last price of 0 is less than {1 - tolerance}, outcome 1 if last price more than {0+tolerance}. \n{num_nas} prices were outside of tolerance.")
    return outcomes


def make_include_column(df):
    '''Make include column for metadata and return'''
    filtered_df = df[df["umaResolutionStatus"] == "resolved"]
    filtered_df = filtered_df.loc[~filtered_df["resolution"].isna()]
    include_idx = df["id"].isin(filtered_df["id"])
    new_df = df
    new_df["include"] = include_idx
    return new_df

# see https://stats.stackexchange.com/questions/161639/does-the-dickey-fuller-test-for-a-random-walk for ADF. If p-val > .05 -> unit root -> random walk
def adf_test(ts):
    '''stastical test for random walk. When P<.05 there is no unit root and timeseries is therefore not a random walk.'''
    test = adfuller(ts, regression = "c")
    dfoutput = pd.Series(
    test[0:4],
    index=["Test Statistic",
        "p-value",
        "#Lags Used",
        "Number of Observations Used",],)
    for key, value in test[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    return dfoutput


def abs_error(p, token_correct):
    abs_error = np.abs(p-token_correct)
    return abs_error


def calc_brier(p, token_correct):
    brier = (p-token_correct)**2
    return brier


def prep_price_data(price_df, meta_data_df):
    '''Takes prices and meta data and gets infers how markets resolved based on last token. 
    Also creates include-column that indicates if market resolved 
    and could resolution could be disambiguated using last trade price
    '''
    meta_data_df["resolution"] = get_outcome(meta_data_df.lastTradePrice, tolerance=0.015)
    meta_data_df = make_include_column(meta_data_df)
    meta_data_df = meta_data_df[["id", "include", "resolution"]]
    price_df = pd.merge(price_df, meta_data_df, left_on = "market_id", right_on="id")
    #make 0-1 indicator column if the token_index matches outcome, ie token was correct. if nan, keep nan
    price_df["token_correct"] = np.where(
        price_df["resolution"].isna(), np.nan,
        (price_df["outcome_index"] == price_df["resolution"]).astype(int))
    return price_df

def main():
    prices = pd.read_csv("data/prices.csv")
    meta_data = pd.read_csv("data/market_metadata.csv")
    prices["datetime"] = pd.to_datetime(prices["datetime"])
    prices = prep_price_data(prices, meta_data)
    prices = prices.assign(brier = calc_brier(p = prices["p"], token_correct=prices["token_correct"]))
    prices = prices.assign(abs_error = abs_error(p = prices["p"], token_correct=prices["token_correct"]))
    prices = prices[prices["outcome_index"] == 1]

    print(f"unique market ids {len(prices['market_id'].unique())}")

    records = []
    for market_id in tqdm(prices["market_id"].unique(), desc="Calculating features for all markets."):
        market = prices[prices["market_id"]==market_id]
        include = market["include"].iloc[0]
        mean_brier = market["brier"].mean()
        mean_abs_error = market["abs_error"].mean()
        market = market.sort_values("datetime")

        #move series to log_odds and get t - t_-1 differences
        ts = market["p"].clip(1e-6, 1 - 1e-6).map(logit)
        residuals = (ts - ts.shift(1)).dropna()
        
        #calc features per market
        abs_drift = np.abs(residuals.mean())
        volatility = residuals.std()
        try:
            adf_res = adf_test(ts)
            adf_test_stat = adf_res["Test Statistic"]
            adf_p_val = adf_res["p-value"]
        except Exception:
            adf_test_stat = np.nan
            adf_p_val = np.nan
        try:
            H, c = compute_Hc(ts, "random_walk")
        except Exception:
            H, c = np.nan, np.nan

        records.append(dict(
            market_id=market_id,
            include=include,
            mean_brier=mean_brier,
            mean_abs_error=mean_abs_error,
            abs_drift=abs_drift,
            volatility=volatility,
            adf_test_stat=adf_test_stat,
            adf_p_val=adf_p_val,
            hurst_exponent=H,
            hurst_c=c,
        ))
    results = pd.DataFrame.from_records(records)
    results.to_csv("data/features.csv", index = False)

if __name__ == "__main__":
    main()


    



