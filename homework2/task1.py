#!/usr/bin/env python

import pickle
import re
import pandas as pd
import requests
from io import StringIO

def save_df(df: pd.DataFrame):
    with open('task1_data.pickle', 'wb') as file:
        pickle.dump(df, file)

def load_df() -> pd.DataFrame:
    with open('task1_data.pickle', 'rb') as file:
        map_data = pickle.load(file)
    return map_data

def get_withdr() -> pd.DataFrame:
    """
    Fetch IPO data for the withdrawn from stockanalysis.com.
    """
    #url = f"https://stockanalysis.com/ipos/{year}/"
    url = f"https://stockanalysis.com/ipos/withdrawn/"
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

def convert_company_class(df: pd.DataFrame) -> pd.DataFrame:
    df['Company Class'] = (df['Company Name']
      .str
      .lower()
      .str
      .split(" ")
      .apply(lambda words: [re.sub(r'[^\w]', '', word) for word in words if word])
      .apply(normalize_company_name)
    )
    return df


def normalize_company_name(names: list[str]) -> str:
    if ("acquisition" in names and "corp" in names) or ("acquisition" in names and "corporation" in names): return "Acq.Corp"
    elif "inc" in names or "incorporated" in names: return "Inc"
    elif "group" in names: return "Group"
    elif "ltd" in names or "limited" in names: return "Limited"
    elif "holdings" in names: return "Holdings"
    else: return "Other"

def define_av_price(df: pd.DataFrame) -> pd.DataFrame:
    df['Avg. price'] = (df['Price Range']
      .apply(convert_price)
    )
    return df

def convert_price(price: str) -> float:
    if price == "-": return None
    elif re.fullmatch(r'^\$\d+\.\d{2}$', price): return [float(num) for num in re.findall(r'^\$(\d+\.\d{2})$', price)][0]
    match = re.search(r'^\$(?P<mini>\d+\.\d{2}) - \$(?P<maxi>\d+\.\d{2})$', price)
    if match: return (float(match.groupdict()['maxi']) + float(match.groupdict()['mini'])) / 2
    print("Some error in price: " + price)
    return None    

def convert_offered(df: pd.DataFrame) -> pd.DataFrame:
    df['Shares Offered'] = pd.to_numeric(df['Shares Offered'], errors='coerce')
    return df

def add_withdrawn(df: pd.DataFrame) -> pd.DataFrame:
    df['Withdrawn Value'] = df['Shares Offered'] * df['Avg. price']
    return df

# 0 Settings
pd.set_option('display.float_format', '{:.2f}'.format)

# 1 LOAD
#ipos_with = get_withdr()
ipos_with = load_df()
#ipos_with.info()

# 2 Company class
ipos_with = convert_company_class(ipos_with)

# 3 Define Avg. price
ipos_with = define_av_price(ipos_with)

# 4 Convert shares offered
ipos_with = convert_offered(ipos_with)

# 5 Withdrawn values
ipos_with = add_withdrawn(ipos_with)

# Check
print(ipos_with[ipos_with['Withdrawn Value'].notna()].shape[0])

# 6 group by company class
sum_with = ipos_with.groupby('Company Class')['Withdrawn Value'].sum()

print(sum_with)

#print(ipos_with.head(10))
