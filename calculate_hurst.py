import numpy as np
import pandas as pd
from hurst import compute_Hc
from tqdm import tqdm

markets = pd.read_csv("data/market_metadata.csv").set_index("id")
markets["start_date"] = pd.to_datetime(markets["startDateIso"])
markets["end_date"] = pd.to_datetime(markets["endDateIso"])
prices = pd.read_csv("data/prices.csv")
prices["datetime"] = pd.to_datetime(prices["datetime"])


def filter_start_end_dates(prices, markets, tolerance_days: int) -> list[int]:
    res = set()
    for market_id, pr in prices.groupby("market_id"):
        m = markets.loc[market_id]
        start_d, end_d = m["start_date"], m["end_date"]
        pr_start, pr_end = pr["datetime"].min(), pr["datetime"].max()
        start_diff = (pr_start - start_d).days
        end_diff = (end_d - pr_end).days
        if (start_diff < tolerance_days) and (end_diff < tolerance_days):
            res.add(market_id)
    return list(res)


def calculate_elapsed(prices, markets, granularity="days"):
    elapsed = []
    for index, row in prices.iterrows():
        e = row["datetime"] - markets.loc[row["market_id"]]["start_date"]
        elapsed.append(getattr(e, granularity))
    return elapsed


def preprocess_timeseries(prices, markets, granularity=None):
    if granularity is not None:
        prices = prices.assign(
            elapsed=calculate_elapsed(prices, markets, granularity=granularity)
        )
    res = {}
    for market_id in tqdm(
        markets.index, total=len(markets.index), desc="preprocessing all markets"
    ):
        timeseries = []
        pr = prices[prices["market_id"] == market_id].sort_values("datetime")
        pr = pr[pr["outcome_index"] == 1]
        if granularity is not None:
            min_elapsed, max_elapsed = pr["elapsed"].min(), pr["elapsed"].max()
            for i_day in range(min_elapsed, max_elapsed):
                day_prices = pr[pr["elapsed"] == i_day]["p"]
                timeseries.append(np.nanmean(day_prices))
        else:
            timeseries = pr["p"]
        res[market_id] = np.array(timeseries)
    return res


timeseries = preprocess_timeseries(prices, markets)
records = []
for market_id in tqdm(timeseries, desc="Calculating hurst exponent for all markets."):
    ts = timeseries[market_id]
    try:
        H, c, data = compute_Hc(timeseries[market_id], "price")
        records.append(dict(market_id=market_id, hurst_exponent=H, c=c))
    except Exception:
        continue

hurst_df = pd.DataFrame.from_records(records)
hurst_df.to_csv("data/hurst_exponent.csv", index=False)
