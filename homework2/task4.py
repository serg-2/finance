#!/usr/bin/env python

import gdown
import pandas as pd

# Get
#file_id = "1grCTCzMZKY5sJRtdbLVCXg8JXA8VPyg-"
#gdown.download(f"https://drive.google.com/uc?id={file_id}", "data.parquet", quiet=False)

# Load
df = pd.read_parquet("data.parquet", engine="pyarrow")

rsi_threshold = 25
selected_df = df[
    (df['rsi'] < rsi_threshold) &
    (df['Date'] >= '2000-01-01') &
    (df['Date'] <= '2025-06-01')
]

net_income = 1000 * (selected_df['growth_future_30d'] - 1).sum()

print(net_income)
