#!/usr/bin/env python
import yfinance as yf

def download_rsi():
    data = yf.download("AMZN", period="1y")
    data['Change'] = data['Close'].diff()

    period = 14
    data['Gain'] = data['Change'].apply(lambda x: x if x > 0 else 0)
    data['Loss'] = data['Change'].apply(lambda x: -x if x < 0 else 0)

    # Средний прирост и убыток
    data['Avg_Gain'] = data['Gain'].rolling(period).mean()
    data['Avg_Loss'] = data['Loss'].rolling(period).mean()
    
    data['RSI'] = 100 - (100 / (1 + (data['Avg_Gain'] / data['Avg_Loss'])))
    return data.dropna()

data = download_rsi()
  
print(data[['Close', 'RSI']])
    