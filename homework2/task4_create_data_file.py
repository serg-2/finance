#!/usr/bin/env python

from datetime import date
import pickle
import time

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader as pdr

from task4_lib import talib_get_momentum_indicators_for_one_ticker, talib_get_pattern_recognition_indicators, talib_get_volume_volatility_cycle_price_indicators



# https://companiesmarketcap.com/usa/largest-companies-in-the-usa-by-market-cap/
US_STOCKS = ['MSFT', 'AAPL', 'GOOG', 'NVDA', 'AMZN', 'META', 'BRK-B', 'LLY', 'AVGO','V', 'JPM']

# You're required to add EU_STOCKS and INDIA_STOCS
# https://companiesmarketcap.com/european-union/largest-companies-in-the-eu-by-market-cap/
EU_STOCKS = ['NVO','MC.PA', 'ASML', 'RMS.PA', 'OR.PA', 'SAP', 'ACN', 'TTE', 'SIE.DE','IDEXY','CDI.PA']

# https://companiesmarketcap.com/india/largest-companies-in-india-by-market-cap/
INDIA_STOCKS = ['RELIANCE.NS','TCS.NS','HDB','BHARTIARTL.NS','IBN','SBIN.NS','LICI.NS','INFY','ITC.NS','HINDUNILVR.NS','LT.NS']
  
ALL_TICKERS = US_STOCKS  + EU_STOCKS + INDIA_STOCKS

def get_parquet_step1() -> pd.DataFrame:
    
  stocks_df = pd.DataFrame({'A' : []})

  for i,ticker in enumerate(ALL_TICKERS):
    print(i,ticker)

    # Work with stock prices
    ticker_obj = yf.Ticker(ticker)

    # historyPrices = yf.download(tickers = ticker,
    #                    period = "max",
    #                    interval = "1d")
    historyPrices = ticker_obj.history(
                       period = "max",
                       interval = "1d")

    # generate features for historical prices, and what we want to predict
    historyPrices['Ticker'] = ticker
    historyPrices['Year']= historyPrices.index.year
    historyPrices['Month'] = historyPrices.index.month
    historyPrices['Weekday'] = historyPrices.index.weekday
    historyPrices['Date'] = historyPrices.index.date

    # historical returns
    for i in [1,3,7,30,90,365]:
      historyPrices['growth_'+str(i)+'d'] = historyPrices['Close'] / historyPrices['Close'].shift(i)
    historyPrices['growth_future_30d'] = historyPrices['Close'].shift(-30) / historyPrices['Close']

    # Technical indicators
    # SimpleMovingAverage 10 days and 20 days
    historyPrices['SMA10']= historyPrices['Close'].rolling(10).mean()
    historyPrices['SMA20']= historyPrices['Close'].rolling(20).mean()
    historyPrices['growing_moving_average'] = np.where(historyPrices['SMA10'] > historyPrices['SMA20'], 1, 0)
    historyPrices['high_minus_low_relative'] = (historyPrices.High - historyPrices.Low) / historyPrices['Close']

    # 30d rolling volatility : https://ycharts.com/glossary/terms/rolling_vol_30
    historyPrices['volatility'] =   historyPrices['growth_1d'].rolling(30).std() * np.sqrt(252)

    # what we want to predict
    historyPrices['is_positive_growth_30d_future'] = np.where(historyPrices['growth_future_30d'] > 1, 1, 0)

    # sleep 1 sec between downloads - not to overload the API server
    time.sleep(.3)
    
    if stocks_df.empty:
      stocks_df = historyPrices
    else:
      stocks_df = pd.concat([stocks_df, historyPrices], ignore_index=True)
  
  stocks_df['ticker_type'] = stocks_df.Ticker.apply(lambda x:get_ticker_type(x, US_STOCKS, EU_STOCKS, INDIA_STOCKS))
  stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
  save_df(stocks_df)

def get_ticker_type(ticker:str, us_stocks_list, eu_stocks_list, india_stocks_list):
  if ticker in us_stocks_list:
    return 'US'
  elif ticker in eu_stocks_list:
    return 'EU'
  elif ticker in india_stocks_list:
    return 'INDIA'
  else:
    return 'ERROR'

def save_df(df: pd.DataFrame):
  with open('task4_step1_data.pickle', 'wb') as file:
      pickle.dump(df, file)
      
def load_step1() -> pd.DataFrame:
    with open('task4_step1_data.pickle', 'rb') as file:
        map_data = pickle.load(file)
    return map_data     

def get_parquet_step2(stocks_df: pd.DataFrame) -> pd.DataFrame:
  stocks_df['Volume'] = stocks_df['Volume']*1.0
  print(stocks_df.info())
  merged_df_with_tech_ind = pd.DataFrame({'A' : []})

  current_ticker_data = None
  i=0
  for ticker in ALL_TICKERS:
    i+=1
    print(f'{i}/{len(ALL_TICKERS)} Current ticker is {ticker}')
    current_ticker_data = stocks_df[stocks_df.Ticker.isin([ticker])]
    # need to have same 'utc' time on both sides
    # https://stackoverflow.com/questions/73964894/you-are-trying-to-merge-on-datetime64ns-utc-and-datetime64ns-columns-if-yo
    current_ticker_data['Date']= pd.to_datetime(current_ticker_data['Date'], utc=True)

    # 3 calls to get additional features
    df_current_ticker_momentum_indicators = talib_get_momentum_indicators_for_one_ticker(current_ticker_data)
    df_current_ticker_momentum_indicators["Date"]= pd.to_datetime(df_current_ticker_momentum_indicators['Date'], utc=True)
    # df_current_ticker_momentum_indicators.loc[:,"Date"]= pd.to_datetime(df_current_ticker_momentum_indicators['Date'], utc=True)

    df_current_ticker_volume_indicators = talib_get_volume_volatility_cycle_price_indicators(current_ticker_data)
    df_current_ticker_volume_indicators["Date"]= pd.to_datetime(df_current_ticker_volume_indicators['Date'], utc=True)
    # df_current_ticker_volume_indicators.loc[:,"Date"]= pd.to_datetime(df_current_ticker_volume_indicators['Date'], utc=True)

    df_current_ticker_pattern_indicators = talib_get_pattern_recognition_indicators(current_ticker_data)
    df_current_ticker_pattern_indicators["Date"]= pd.to_datetime(df_current_ticker_pattern_indicators['Date'], utc=True)
    # df_current_ticker_pattern_indicators.loc[:,"Date"]= pd.to_datetime(df_current_ticker_pattern_indicators['Date'], utc=True)

    # merge to one df
    m1 = pd.merge(current_ticker_data, df_current_ticker_momentum_indicators.reset_index(), how = 'left', on = ["Date","Ticker"], validate = "one_to_one")
    m2 = pd.merge(m1, df_current_ticker_volume_indicators.reset_index(), how = 'left', on = ["Date","Ticker"], validate = "one_to_one")
    m3 = pd.merge(m2, df_current_ticker_pattern_indicators.reset_index(), how = 'left', on = ["Date","Ticker"], validate = "one_to_one")

    if merged_df_with_tech_ind.empty:
      merged_df_with_tech_ind = m3
    else:
      merged_df_with_tech_ind = pd.concat([merged_df_with_tech_ind,m3], ignore_index = False)

  # remove timezone
  merged_df_with_tech_ind['Date'] = pd.to_datetime(merged_df_with_tech_ind['Date']).dt.tz_localize(None)

  return merged_df_with_tech_ind


def get_growth_df(df:pd.DataFrame, prefix:str)->pd.DataFrame:
  for i in [1,3,7,30,90,365]:
    #DEBUG: dax_daily['Adj Close_sh_m_'+str(i)+'d'] = dax_daily['Adj Close'].shift(i)
    df['growth_'+prefix+'_'+str(i)+'d'] = df['Close'] / df['Close'].shift(i)
    GROWTH_KEYS = [k for k in df.keys() if k.startswith('growth')]
  return df[GROWTH_KEYS]

def get_ticker_to_merge(ticker: str, prefix: str) -> pd.DataFrame:
  ticker_obj = yf.Ticker(ticker)
  tickerdaily = ticker_obj.history(period = "max", interval = "1d")
  ticker_to_merge = get_growth_df(tickerdaily, prefix)
  ticker_to_merge.index = ticker_to_merge.index.tz_localize(None)  
  return ticker_to_merge


def get_gdp_pot() -> pd.DataFrame:
  end = date.today()
  start = date(year=end.year-70, month=end.month, day=end.day)
  # Real Potential Gross Domestic Product (GDPPOT), Billions of Chained 2012 Dollars, QUARTERLY
  # https://fred.stlouisfed.org/series/GDPPOT
  gdppot = pdr.DataReader("GDPPOT", "fred", start=start)
  gdppot['gdppot_us_yoy'] = gdppot.GDPPOT/gdppot.GDPPOT.shift(4)-1
  gdppot['gdppot_us_qoq'] = gdppot.GDPPOT/gdppot.GDPPOT.shift(1)-1
  return gdppot[['gdppot_us_yoy','gdppot_us_qoq']]

def get_core_cpi() -> pd.DataFrame:
  end = date.today()
  start = date(year=end.year-70, month=end.month, day=end.day)
  # # "Core CPI index", MONTHLY
  # https://fred.stlouisfed.org/series/CPILFESL
  # The "Consumer Price Index for All Urban Consumers: All Items Less Food & Energy"
  # is an aggregate of prices paid by urban consumers for a typical basket of goods, excluding food and energy.
  # This measurement, known as "Core CPI," is widely used by economists because food and energy have very volatile prices.
  cpilfesl = pdr.DataReader("CPILFESL", "fred", start=start)
  cpilfesl['cpi_core_yoy'] = cpilfesl.CPILFESL/cpilfesl.CPILFESL.shift(12)-1
  cpilfesl['cpi_core_mom'] = cpilfesl.CPILFESL/cpilfesl.CPILFESL.shift(1)-1
  return cpilfesl[['cpi_core_yoy','cpi_core_mom']]

def get_fed_funds() -> pd.DataFrame:
  end = date.today()
  start = date(year=end.year-70, month=end.month, day=end.day)
  # Fed rate https://fred.stlouisfed.org/series/FEDFUNDS
  return pdr.DataReader("FEDFUNDS", "fred", start=start)

def get_dgs(year_def: str) -> pd.DataFrame:
  end = date.today()
  start = date(year=end.year-70, month=end.month, day=end.day)
  # https://fred.stlouisfed.org/series/DGS1
  # https://fred.stlouisfed.org/series/DGS5
  # https://fred.stlouisfed.org/series/DGS10
  return pdr.DataReader("DGS" + year_def, "fred", start=start)

def get_vix() -> pd.DataFrame:
  # VIX - Volatility Index
  # https://finance.yahoo.com/quote/%5EVIX/

  ticker_obj = yf.Ticker("^VIX")
  vix = ticker_obj.history(
                     period = "max",
                     interval = "1d")
  vix_to_merge = vix['Close']
  vix_to_merge.index = vix_to_merge.index.tz_localize(None)
  return vix_to_merge 





# Get
# import gdown
# file_id = "1grCTCzMZKY5sJRtdbLVCXg8JXA8VPyg-"
# gdown.download(f"https://drive.google.com/uc?id={file_id}", "data.parquet", quiet=False)
# OR
#  
# CREATE


# Load OR GET
#get_parquet_step1()
df = load_step1()

m1 = get_parquet_step2(df)

# Add Dax daily
m2 = pd.merge(m1,
              get_ticker_to_merge("^GDAXI", "dax"),
              how='left',
              left_on='Date',
              right_index=True,
              validate = "many_to_one"
              )

# Add snp500 daily
m3 = pd.merge(m2,
              get_ticker_to_merge("^GSPC", "snp500"),
              how='left',
              left_on='Date',
              right_index=True,
              validate = "many_to_one"
              )

# Add dji daily
m4 = pd.merge(m3,
              get_ticker_to_merge("^DJI", "dji"),
              how='left',
              left_on='Date',
              right_index=True,
              validate = "many_to_one"
              )

# Add epi_etf
m5 = pd.merge(m4,
              get_ticker_to_merge("EPI", "epi"),
              how='left',
              left_on='Date',
              right_index=True,
              validate = "many_to_one"
              )

# define quarter as the first date of qtr
m5['Quarter'] = m5['Date'].dt.to_period('Q').dt.to_timestamp()

m6 = pd.merge(m5,
              get_gdp_pot(),
              how='left',
              left_on='Quarter',
              right_index=True,
              validate = "many_to_one"
              )

m6['Month'] = m6['Date'].dt.to_period('M').dt.to_timestamp()

m7 = pd.merge(m6,
              get_core_cpi(),
              how='left',
              left_on='Month',
              right_index=True,
              validate = "many_to_one"
              )

# PROBLEM! Last month is not defined
fields_to_fill = ['cpi_core_yoy',	'cpi_core_mom']
# Fill missing values in selected fields with the last defined value
for field in fields_to_fill:
  m7[field] = m7[field].ffill()

m8 = pd.merge(m7,
              get_fed_funds(),
              how='left',
              left_on='Month',
              right_index=True,
              validate = "many_to_one"
              )

fields_to_fill = ['FEDFUNDS']
# Fill missing values in selected fields with the last defined value
for field in fields_to_fill:
  m8[field] = m8[field].ffill()

m9 = pd.merge(m8,
              get_dgs("1"),
              how='left',
              left_on='Date',
              right_index=True,
              validate = "many_to_one"
              )

m10 = pd.merge(m9,
              get_dgs("5"),
              how='left',
              left_on='Date',
              right_index=True,
              validate = "many_to_one"
              )

m11 = pd.merge(m10,
              get_dgs("10"),
              how='left',
              left_on='Date',
              right_index=True,
              validate = "many_to_one"
              )

m12 = pd.merge(m11,
              get_vix(),
              how='left',
              left_on='Date',
              right_index=True,
              validate = "many_to_one"
              )

# Add Gold
m13 = pd.merge(m12,
              get_ticker_to_merge("GC=F", "gold"),
              how='left',
              left_on='Date',
              right_index=True,
              validate = "many_to_one"
              )

# Crude oil
m14 = pd.merge(m13,
              get_ticker_to_merge("CL=F", "wti_oil"),
              how='left',
              left_on='Date',
              right_index=True,
              validate = "many_to_one"
              )

# Brent oil
m15 = pd.merge(m14,
              get_ticker_to_merge("BZ=F", "brent_oil"),
              how='left',
              left_on='Date',
              right_index=True,
              validate = "many_to_one"
              )

# Btc to USD
m16 = pd.merge(m15,
              get_ticker_to_merge("BTC-USD", "btc_usd"),
              how='left',
              left_on='Date',
              right_index=True,
              validate = "many_to_one"
              )

fields_to_fill = ['gdppot_us_yoy','gdppot_us_qoq','cpi_core_yoy','cpi_core_mom','FEDFUNDS','DGS1','DGS5','DGS10']

# Fill missing values in selected fields with the last defined value
for field in fields_to_fill:
    m16[field] = m16[field].ffill()

#print(m16.info())
d = m16.Date.max()

m16.to_parquet(f'stocks_df_combined_{d.strftime('%Y_%m_%d')}.parquet.brotli',
              compression='brotli')
