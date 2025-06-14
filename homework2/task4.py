#!/usr/bin/env python

from io import StringIO

import pandas as pd
import requests

pd.options.mode.chained_assignment = None  # default='warn'

def get_ipos_by_year(year: int) -> pd.DataFrame:
    """
    Fetch IPO data for the given year from stockanalysis.com.
    """
    url = f"https://stockanalysis.com/ipos/{year}/"
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/58.0.3029.110 Safari/537.3'
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Wrap HTML text in StringIO to avoid deprecation warning
        # "Passing literal html to 'read_html' is deprecated and will be removed in a future version. To read from a literal string, wrap it in a 'StringIO' object."
        html_io = StringIO(response.text)
        tables = pd.read_html(html_io)

        if not tables:
            raise ValueError(f"No tables found for year {year}.")

        return tables[0]

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except ValueError as ve:
        print(f"Data error: {ve}")
    except Exception as ex:
        print(f"Unexpected error: {ex}")

    return pd.DataFrame()


### MAIN From downloaded info ==================================================

df = pd.read_parquet("data.parquet", engine="pyarrow")
#df.info()
#Index: 229932 entries, 0 to 5690
#Columns: 203 entries, Open to growth_btc_usd_365d
#dtypes: datetime64[ns](3), float64(129), int32(64), int64(5), object(2)
#memory usage: 301.7+ MB

rsi_threshold = 25
selected_df = df[
    (df['rsi'] < rsi_threshold) &
    (df['Date'] >= '2000-01-01') &
    (df['Date'] <= '2025-06-01')
]

net_income = 1000 * (selected_df['growth_future_30d'] - 1).sum()

print(net_income)
