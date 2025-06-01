#!/usr/bin/env python
import yfinance as yf
from datetime import date

# MAIN

def getPercentage(name):
    ticker_obj = yf.Ticker(name)

    start = date(year=2025, month=1, day=1)
    end = date(year=2025, month=5, day=1)

    dax_daily = ticker_obj.history(start = start, end = end)

    s = dax_daily.iloc[:,0].head(1).iloc[0]
    f = dax_daily.iloc[:,0].tail(1).iloc[0]

    return name, ((f - s) / s) * 100


print (getPercentage("^GSPC"))
print (getPercentage("000001.SS"))
print (getPercentage("^HSI"))
print (getPercentage("^AXJO"))
print (getPercentage("^NSEI"))
print (getPercentage("^GSPTSE"))
print (getPercentage("^GDAXI"))
print (getPercentage("^FTSE"))
print (getPercentage("^N225"))
print (getPercentage("^MXX"))
print (getPercentage("^BVSP"))


