import numpy as np
import pandas as pd
from tqdm import tqdm

from hurst import compute_Hc
from calculate_brier import prep_price_data, calc_brier

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

    #filter for outcome _index 1
    prices = prices[prices["outcome_index"] == 1]
    
    broken_hurst = 0
    records = []
    for market_id in tqdm(prices["market_id"].unique(), desc="Calculating features for all markets."):
        
        #get market
        market = prices[prices["market_id"]==market_id]
        include = market["include"].iloc[0]

        #get mean brier
        mean_brier = market["brier"].mean()

        #make timeseries for hurst
        market.sort_values("datetime")
        ts = market["p"].to_numpy()

        

        #try hurst 
        try:
            H, c, data = compute_Hc(ts, "price")
            records.append(dict(market_id=market_id, include = include, hurst_exponent=H, hurst_c=c, mean_brier=mean_brier))
        except Exception:
            records.append(dict(market_id=market_id, include = include, hurst_exponent=np.nan, hurst_c=np.nan, mean_brier=mean_brier))
            broken_hurst += 1
            continue
    
    print(f"Hurst calculation failed for {broken_hurst} markets")

    results = pd.DataFrame.from_records(records)
    results.to_csv("data/features.csv", index = False)

    



