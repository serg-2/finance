#!/usr/bin/env python

import pickle
from datetime import date
import numpy as np
import pandas as pd
import requests
from io import StringIO
import yfinance as yf

def save_df(df: pd.DataFrame):
    with open('task3_data.pickle', 'wb') as file:
        pickle.dump(df, file)

def load_df() -> pd.DataFrame:
    with open('task2_data.pickle', 'rb') as file:
        map_data = pickle.load(file)
    return map_data

def filter_date(df: pd.DataFrame) -> pd.DataFrame:
    # Filter empty
    df2 = df[~df['IPO Price'].str.contains('-')]
    df2['IPO Date'] = pd.to_datetime(df2['IPO Date'])
    return df2[df2['IPO Date'].between('2023-01-01', '2024-06-01')]

def get_ticker(ticker: str) -> pd.DataFrame:
  ticker_obj = yf.Ticker(ticker)

  historyPrices = ticker_obj.history(
                     start = date(year=2024, month=1, day=1),
                     end = date(year=2025, month=6, day=7),
                     interval = "1d"
                     )
  
  historyPrices['Ticker'] = ticker

  for i in [1,2,3,4,5,6,7,8,9,10,11,12]:
    historyPrices['future_growth_'+str(i)+'m'] = historyPrices['Close'].shift(i * -21) / historyPrices['Close'] 
  
  historyPrices['min_date'] = historyPrices.index.min()
  #print("MIN DATE: ", historyPrices.index.min())
  historyPrices = historyPrices.loc[historyPrices.index == historyPrices['min_date']]

  return historyPrices

def get_stocks(tickers: list[str]) -> None:
  stocks_df = pd.DataFrame({'A' : []})
  for i,ticker in enumerate(tickers):
    #if i != 0: continue
    if stocks_df.empty:
      stocks_df = get_ticker(ticker)
    else:
      stocks_df = pd.concat([stocks_df, get_ticker(ticker)], ignore_index=True)
  
  with open('task3_data.pickle', 'wb') as file:
        pickle.dump(stocks_df, file)

def load_stocks() -> pd.DataFrame:
   with open('task3_data.pickle', 'rb') as file:
        map_data = pickle.load(file)
   return map_data

# 0 Settings
pd.set_option('display.float_format', '{:.2f}'.format)

# 1 LOAD and filter date
ipos_with = load_df()
ipos_with = filter_date(ipos_with)
tickers = ipos_with.iloc[:,[1]].values.flatten()
print("Number of tickers: ", tickers.size)

#get_stocks(tickers)
stocks_df = load_stocks()

# describe
for i in [1,2,3,4,5,6,7,8,9,10,11,12]:
    print("Months: ", i,stocks_df['future_growth_'+str(i)+'m'].describe()['mean'])
    

#print(stocks_df)
#print(stocks_df.columns)
