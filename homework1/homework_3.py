#!/usr/bin/env python
import yfinance as yf
from datetime import date
import numpy as np
import pandas as pd

def getHigh():
    ticker_obj = yf.Ticker("^GSPC")

    start = date(year=1950, month=1, day=1)
    dax_daily = ticker_obj.history(start = start)

    #s = dax_daily.iloc[:,[1,2]].tail(250)
    s = dax_daily.iloc[:,[3]]

    max = 0
    loc_min = 0
    maxDate = 0
    draws = {}

    for index, row in enumerate(s.iterrows()):
        if index == 0: 
            max = row[1].iloc[0]
            loc_min = row[1].iloc[0]
            maxDate = row[0]
            continue
        
        if row[1].iloc[0] > max:
            # NEW MAX VALUE!
            max = row[1].iloc[0]
            maxDate = row[0]

            # Reset Min
            loc_min = row[1].iloc[0]
        else:
            if row[1].iloc[0] < loc_min:
                loc_min = row[1].iloc[0]
                drawdown = ((max - loc_min ) / max ) * 100
                draws[maxDate] = (drawdown, (row[0]- maxDate).days) 
    return draws

def filter_drawbacks(draws, filter_value):
    result = {}
    for key,value in draws.items():
        if value[0] > filter_value:
            result[key] = value
    return result 

def get_duration(draws):
    result = []
    for _,value in draws.items():
        result.append(value[1])
    return result

def calculate_percentiles(data):
    percentiles = {
        '25th': np.percentile(data, 25),
        '50th (median)': np.percentile(data, 50),
        '75th': np.percentile(data, 75)
    }
    return percentiles

drawbacks = getHigh()
filtered_drawbacks = filter_drawbacks(drawbacks, 5)
durations = get_duration(filtered_drawbacks)

print (calculate_percentiles(durations))

