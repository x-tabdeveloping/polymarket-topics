import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


def parse_entries(s: str) -> list[str]:
    s = s[s.find("{") :]
    bracket_counter = 0
    entries = []
    entry = ""
    for i, character in enumerate(s):
        entry += character
        entry = entry.strip()
        if character == "{":
            bracket_counter += 1
        elif character == "}":
            bracket_counter -= 1
        if (bracket_counter == 0) and (len(entry) >= 2):
            entries.append(entry)
            entry = ""
    return entries


def main():
    data_dir = Path("data/Polymarket_dataset/Polymarket_dataset/")
    records = []
    print("Collecting price records")
    for market_dir in data_dir.glob("market=*"):
        for token_file in market_dir.joinpath("price").glob("token=*"):
            with token_file.open() as in_file:
                entries = parse_entries(in_file.read())
                records.extend(map(json.loads, entries))
    prices = pd.DataFrame.from_records(records)
    print("Converting timestamps")
    prices["datetime"] = prices["t"].map(datetime.fromtimestamp)
    print("Saving prices.")
    prices.to_csv("data/prices.csv", index=False)
    print("Collecting metadata for all markets.")
    unique_markets = prices["market_id"].unique()
    market_metadata_entries = []
    for market_id in tqdm(unique_markets, "Getting metadata for all markets."):
        status = -1
        n_retries = 0
        while status != 200:
            if n_retries > 4:
                warnings.warn(
                    f"Request couldn't be completed after {n_retries} retries."
                )
                break
            r = requests.get(f"https://gamma-api.polymarket.com/markets/{market_id}")
            status = r.status_code
            if status == 200:
                market_metadata_entries.append(r.json())
            else:
                warnings.warn(
                    f"Request unsuccessful, code: {status}, waiting 5 seconds..."
                )
                time.sleep(5)
            n_retries += 1
    print("Saving.")
    market_metadata = pd.DataFrame.from_records(market_metadata_entries)
    market_metadata.to_csv("data/market_metadata.csv", index=False)
    print("DONE")


if __name__ == "__main__":
    main()
