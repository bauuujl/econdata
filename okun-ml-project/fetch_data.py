"""Download real GDP growth (annual %) for every country from the World Bank API.

The repo's existing GDP series is nominal US$, which mixes real growth with
inflation and exchange-rate moves. Okun's law needs *real* growth, so we pull
indicator NY.GDP.MKTP.KD.ZG once and save it to data/real_gdp_growth.csv.
"""
import json
import urllib.request
from pathlib import Path

import pandas as pd

URL = ("https://api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.KD.ZG"
       "?format=json&date=1990:2024&per_page=20000")
OUT = Path(__file__).parent / "data" / "real_gdp_growth.csv"


def main():
    with urllib.request.urlopen(URL, timeout=60) as resp:
        meta, rows = json.load(resp)
    df = pd.DataFrame(
        {
            "country_code": r["countryiso3code"],
            "year": int(r["date"]),
            "real_gdp_growth": r["value"],
        }
        for r in rows
        if r["value"] is not None and r["countryiso3code"]
    )
    OUT.parent.mkdir(exist_ok=True)
    df.sort_values(["country_code", "year"]).to_csv(OUT, index=False)
    print(f"Saved {len(df):,} rows to {OUT}")


if __name__ == "__main__":
    main()
