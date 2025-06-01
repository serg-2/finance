#!/usr/bin/env python

import pandas as pd

# URL страницы Википедии со списком компаний S&P 500
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Используем pandas.read_html для чтения всех таблиц с страницы
tables = pd.read_html(url)

# Первая таблица на странице содержит список компаний S&P 500
sp500_table = tables[0]

column_name = sp500_table.columns[5]
sp500_table[column_name] = sp500_table[column_name].str.split("-").str[0]

edited_table = sp500_table.iloc[:,[0,1,5]]

sector = edited_table.groupby(column_name).size()

print(sp500_table.head(10))
print(sector)