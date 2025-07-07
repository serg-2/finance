#!/usr/bin/env python

import numpy as np
import pandas as pd
import pickle
from homework3_lib import get_numerical


def save_df(df: pd.DataFrame, filename: str):
    with open(filename, 'wb') as file:
        pickle.dump(df, file)

def load_df(filename: str) -> pd.DataFrame:
    with open(filename, 'rb') as file:
        map_data = pickle.load(file)
    return map_data

def load_data() -> pd.DataFrame:
    # Downloading data
    # https://drive.google.com/uc?id=1mb0ae2M5AouSDlqcUnIwaHq7avwGNrmB

    df_full = pd.read_parquet("./stocks_df_combined_2025_06_13.parquet.brotli", )
    #df_full.info()
    #print(df_full.keys())

    # growth indicators (but not future growth)
    GROWTH = [g for g in df_full.keys() if (g.find('growth_')==0)&(g.find('future')<0)]
    OHLCV = ['Open','High','Low','Close','Adj Close_x','Volume']
    CATEGORICAL = ['Month', 'Weekday', 'Ticker', 'ticker_type', 'month_wom']
    CATEGORICAL_TASK2 = ['Month', 'Weekday', 'Ticker', 'ticker_type']
    TO_PREDICT = [g for g in df_full.keys() if (g.find('future')>=0)]
    #print(TO_PREDICT)
    TO_DROP = ['Year','Date','index_x', 'index_y', 'index', 'Quarter','Adj Close_y'] + CATEGORICAL + OHLCV

    # let's define on more custom numerical features
    df_full['ln_volume'] = df_full.Volume.apply(lambda x: np.log(x))

    NUMERICAL = get_numerical(df_full)

    # CHECK: NO OTHER INDICATORS LEFT
    OTHER = [k for k in df_full.keys() if k not in OHLCV + CATEGORICAL + NUMERICAL + TO_DROP]

    print(df_full.Ticker.nunique())

    # tickers, min-max date, count of daily observations
    df_full.groupby(['Ticker'])['Date'].agg(['min','max','count'])

    # truncated df_full with 25 years of data (and defined growth variables)
    df = df_full[df_full.Date>='2000-01-01']
    df.info()
    #print(df.keys().tolist())

    # let look at the features count and df size:
    df[NUMERICAL].info()

    ###### Generating dummies =====================================

    print(CATEGORICAL)

    # dummy variables are not generated from Date and numeric variables
    df.loc[:,'Month'] = df.Month.dt.strftime('%B')
    df.loc[:,'Weekday'] = df.Weekday.astype(str)
    df.loc[:,'DOM'] = df.Date.dt.day.values
    df.loc[:,'WOM'] = (df.DOM-1) // 7 + 1
    df.loc[:,'month_wom'] = df.apply(lambda x: f"{x['Month']}_w{x['WOM']}", axis=1)
    df.drop('DOM', axis=1, inplace=True)
    df.drop('WOM', axis=1, inplace=True)

    #print(df.month_wom.tail(5))

    # Generate dummy variables (no need for bool, let's have int32 instead)
    # Task 1
    dummy_variables = pd.get_dummies(df[CATEGORICAL], dtype='int32')
    # Task2
    #dummy_variables = pd.get_dummies(df[CATEGORICAL_TASK2], dtype='int32')

    dummy_variables.info()

    # get dummies names in a list
    DUMMIES = dummy_variables.keys().to_list()

    # Concatenate the dummy variables with the original DataFrame
    df_with_dummies = pd.concat([df, dummy_variables], axis=1)
    save_df(df_with_dummies, 'dummies.pickle')

    df_with_dummies[NUMERICAL+DUMMIES].info()

    corr_is_positive_growth_30d_future = df_with_dummies[NUMERICAL+DUMMIES+TO_PREDICT].corr()['is_positive_growth_30d_future']

    # create a dataframe for an easy way to sort
    corr_is_positive_growth_30d_future_df = pd.DataFrame(corr_is_positive_growth_30d_future)
    return corr_is_positive_growth_30d_future_df[corr_is_positive_growth_30d_future_df.index.str.startswith('month_wom')].sort_values(by='is_positive_growth_30d_future')

#1 Load data from file

save_df(load_data(), 'task1_data.pickle')
df = load_df('task1_data.pickle')
print(df.tail(1))
