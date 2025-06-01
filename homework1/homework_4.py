#!/usr/bin/env python
import yfinance as yf
from datetime import date,timedelta,datetime
import numpy as np
import pandas as pd

import pickle

def update_cache():
    result ={}
    ticker_obj = yf.Ticker("AMZN")

    start = date(year=1950, month=1, day=1)
    dax_daily = ticker_obj.history(start = start)

    #s = dax_daily.iloc[:,[1,2]].tail(250)
    #s = dax_daily.iloc[:,[3]]
    s = dax_daily.iloc[:,[3]]

    old_date = 0

    start_index = 0

    for index, row in enumerate(s.iterrows()):
        if index == start_index: 
            day1 = row[1].iloc[0]
            continue
        if index == start_index + 1:
            day2 = row[1].iloc[0]
            old_date = row[0].date()
            continue
        if index == start_index + 2:
            day3 = row[1].iloc[0]
        else:
            if (row[0].date() - old_date).days > 1:
                start_index = index - 1
                continue
            day1 = day2
            day2 = day3
            day3 = row[1].iloc[0]

        change = day3/day1 -1
        result[old_date] = change * 100

        old_date = row[0].date()
    with open('tmp_data.pickle', 'wb') as file:
        pickle.dump(result, file)

def get_yf_cache():
    with open('tmp_data.pickle', 'rb') as file:
        map_data = pickle.load(file)
    return map_data

def read_csv():
    result = {}
    a =pd.read_csv("ha1_Amazon.csv", delimiter=';')
    b = a.iloc[:,[2,5]]
    for index, row in enumerate(b.iterrows()):
        if not isinstance(row[1].iloc[0], str): continue
        if row[1].iloc[1] == "-": continue
        timestamp = pd.to_datetime(row[1].iloc[0].replace("EDT","-0400").replace("EST","-0500"))
        
        val = float(row[1].iloc[1])
        if val > 0 and timestamp.date() < datetime.now().date():
            result[timestamp.date()] = val
    return result

update_cache()

yfin_data = get_yf_cache()

vals = []

for key,value in read_csv().items():
    val_a = yfin_data.get(key + timedelta(days=1))
    if val_a is not None:
        print("Date: ",key, " value:",val_a)
        vals.append(val_a)
    else:
        print("No val")

print("median: ", np.percentile(vals, 50))
