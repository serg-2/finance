#!/usr/bin/env python

import pickle
from datetime import date
import numpy as np
import pandas as pd
import requests
from io import StringIO
import yfinance as yf

def save_df(df: pd.DataFrame):
    with open('task2_data.pickle', 'wb') as file:
        pickle.dump(df, file)

def load_df() -> pd.DataFrame:
    with open('task2_data.pickle', 'rb') as file:
        map_data = pickle.load(file)
    return map_data

def get_withdr() -> pd.DataFrame:
    """
    Fetch IPO data for the withdrawn from stockanalysis.com.
    """
    url = f"https://stockanalysis.com/ipos/2024/"
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
            raise ValueError(f"No tables found for withdrawn.")

        #Save to cache
        save_df(tables[0])
        return tables[0]

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except ValueError as ve:
        print(f"Data error: {ve}")
    except Exception as ex:
        print(f"Unexpected error: {ex}")

    return pd.DataFrame()

def filter_date(df: pd.DataFrame) -> pd.DataFrame:
    # Filter empty
    df2 = df[~df['IPO Price'].str.contains('-')]
    df2['IPO Date'] = pd.to_datetime(df2['IPO Date'])
    return df2[df2['IPO Date'].between('2023-01-01', '2024-06-01')]

def get_ticker(ticker: str) -> pd.DataFrame:
  ticker_obj = yf.Ticker(ticker)

  historyPrices = ticker_obj.history(
                     end = date(year=2025, month=6, day=7),
                     start = date(year=2024, month=6, day=1),
                     interval = "1d"
                     )
  
  historyPrices['Ticker'] = ticker
  historyPrices['growth_252d'] = historyPrices['Close'] / historyPrices['Close'].shift(252)
  historyPrices['growth_1d'] = historyPrices['Close'] / historyPrices['Close'].shift(1)
  historyPrices['volatility'] = historyPrices['growth_1d'].rolling(30).std() * np.sqrt(252)
  historyPrices['Sharpe'] = (historyPrices['growth_252d'] - 0.045) / historyPrices['volatility']
  historyPrices = historyPrices.tail(1)
  return historyPrices

def get_stocks(tickers: list[str]) -> None:
  stocks_df = pd.DataFrame({'A' : []})
  for i,ticker in enumerate(tickers):
    if stocks_df.empty:
      stocks_df = get_ticker(ticker)
    else:
      stocks_df = pd.concat([stocks_df, get_ticker(ticker)], ignore_index=True)
  
  with open('task2_2_data.pickle', 'wb') as file:
        pickle.dump(stocks_df, file)

def load_stocks() -> pd.DataFrame:
   with open('task2_2_data.pickle', 'rb') as file:
        map_data = pickle.load(file)
   return map_data

# 0 Settings
pd.set_option('display.float_format', '{:.2f}'.format)

# 1 LOAD and filter date
#ipos_with = get_withdr()
ipos_with = load_df()
#ipos_with.info()

ipos_with = filter_date(ipos_with)
# Check
print(ipos_with.shape[0])

# 2 daily stocks from yfinance
tickers = ipos_with.iloc[:,[1]].values.flatten()

#get_stocks(tickers)
stocks_df = load_stocks()

print(stocks_df['growth_252d'].describe())
print(stocks_df['Sharpe'].describe())

# Additional
print("Additional Task ====================================")
print(stocks_df.sort_values('growth_252d', na_position='first').tail(10))
print(stocks_df.sort_values('Sharpe', na_position='first').tail(10))
