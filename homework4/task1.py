# IMPORTS
import time
import numpy as np
import pandas as pd
from joblib import dump, load

#Fin Data Sources
import yfinance as yf
# import pandas_datareader as pdr

#Data viz
import plotly.graph_objs as go
import plotly.graph_objects as go
import plotly.express as px

import time
from datetime import date

# for graphs
import matplotlib.pyplot as plt
import seaborn as sns

# Imports form ML (Decision Trees)
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

# Disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'


# Original
#CATEGORICAL = ['Month', 'Weekday', 'Ticker', 'ticker_type']
CATEGORICAL = ['Month', 'Weekday', 'ticker_type']


# All Supported Ta-lib indicators: https://github.com/TA-Lib/ta-lib-python/blob/master/docs/funcs.md
TECHNICAL_INDICATORS = ['adx', 'adxr', 'apo', 'aroon_1','aroon_2', 'aroonosc',
     'bop', 'cci', 'cmo','dx', 'macd', 'macdsignal', 'macdhist', 'macd_ext',
     'macdsignal_ext', 'macdhist_ext', 'macd_fix', 'macdsignal_fix',
     'macdhist_fix', 'mfi', 'minus_di', 'mom', 'plus_di', 'dm', 'ppo',
     'roc', 'rocp', 'rocr', 'rocr100', 'rsi', 'slowk', 'slowd', 'fastk',
     'fastd', 'fastk_rsi', 'fastd_rsi', 'trix', 'ultosc', 'willr',
     'ad', 'adosc', 'obv', 'atr', 'natr', 'ht_dcperiod', 'ht_dcphase',
     'ht_phasor_inphase', 'ht_phasor_quadrature', 'ht_sine_sine', 'ht_sine_leadsine',
     'ht_trendmod', 'avgprice', 'medprice', 'typprice', 'wclprice']

MACRO = ['gdppot_us_yoy', 'gdppot_us_qoq', 'cpi_core_yoy', 'cpi_core_mom', 'FEDFUNDS',
     'DGS1', 'DGS5', 'DGS10']

# manually defined features
CUSTOM_NUMERICAL = ['SMA10', 'SMA20', 'growing_moving_average', 'high_minus_low_relative','volatility', 'ln_volume','div_payout','stock_split']
    
def GET_GROWTH(df_full: pd.DataFrame) -> list:
    return [g for g in df_full.keys() if (g.find('growth_')==0)&(g.find('future')<0)]

def GET_TECHNICAL_PATTERNS(df_full: pd.DataFrame) -> list:
    return [g for g in df_full.keys() if g.find('cdl')>=0]

def GET_NUMERICAL(df: pd.DataFrame) -> list:
    return GET_GROWTH(df) + TECHNICAL_INDICATORS + GET_TECHNICAL_PATTERNS(df) + CUSTOM_NUMERICAL + MACRO

def GET_TO_PREDICT(df_full: pd.DataFrame) -> list:
    return [g for g in df_full.keys() if (g.find('future')>=0)]

def GET_DUMMIES(df: pd.DataFrame) -> list:
    # Generate dummy variables (no need for bool, let's have int32 instead)
    dummy_variables = pd.get_dummies(df[CATEGORICAL], dtype='int32')
    #print(dummy_variables.info())
    # get dummies names in a list
    return dummy_variables.keys().to_list()

import pickle
def save_df(df: pd.DataFrame, filename: str):
    with open(filename, 'wb') as file:
        pickle.dump(df, file)

def load_df(filename: str) -> pd.DataFrame:
    with open(filename, 'rb') as file:
        map_data = pickle.load(file)
    return map_data

def save_tree(clf: DecisionTreeClassifier, filename: str):
    # Save the classifier to a file named 'decision_tree_model.joblib'
    dump(clf, filename)

def load_tree(filename: str) -> DecisionTreeClassifier:
    return load(filename)

def create_step_1():
    # Load from 
    # https://drive.google.com/uc?id=1mb0ae2M5AouSDlqcUnIwaHq7avwGNrmB

    # full dataset for 33 stocks
    df_full = pd.read_parquet("./stocks_df_combined_2025_06_13.parquet.brotli", )

    #df_full.info()
    

    #print(df_full.keys())

    # leaving only Volume ==> generate ln(Volume)
    OHLCV = ['Open','High','Low','Close_x','Volume']

    # we define dummy variables on Dividends and Stock Splits events later, but drop the original abs. values
    TO_DROP = ['Year','Date','index_x', 'index_y', 'index', 'Quarter','Close_y','Dividends','Stock Splits'] + CATEGORICAL + OHLCV
    #print(TO_DROP)

    # let's define on more custom numerical features
    # Add a small constant to avoid log(0)
    df_full['ln_volume'] = df_full.Volume.apply(lambda x: np.log(x+ 1e-6))

    # define columns on Dividends or Stock Splits
    df_full['div_payout'] = (df_full.Dividends>0).astype(int)
    df_full['stock_split'] = (df_full['Stock Splits']>0).astype(int)
    
    # CHECK: NO OTHER INDICATORS LEFT
    OTHER = [k for k in df_full.keys() if k not in OHLCV + CATEGORICAL + GET_NUMERICAL(df_full) + TO_DROP + GET_TO_PREDICT(df_full)]
    #print(OTHER)

    #print(df_full.Ticker.nunique())

    # truncated df_full with 25 years of data (and defined growth variables)
    df = df_full[df_full.Date>='2000-01-01']
    #print(df.info())
    
    save_df(df, 'step1.pickle')

def load_step_1() -> pd.DataFrame:
    return load_df("step1.pickle")

#create_step_1()
#df = load_step_1()

def create_step_2(df: pd.DataFrame) -> pd.DataFrame:

    # tickers, min-max date, count of daily observations
    #print(df.groupby(['Ticker'])['Date'].agg(['min','max','count']))

    #print(CATEGORICAL)

    # dummy variables are not generated from Date and numeric variables

    # df.loc[:,'Month'] = df['Month'].dt.strftime('%B').astype('string')

    df.loc[:,'Month']= pd.to_datetime(df['Month'], format='%m').dt.strftime('%B')

    df.loc[:,'Weekday'] = df['Weekday'].astype('string')
    # .astype(str)

    # Task 1 -remove
    # define week of month
    #df.loc[:,'wom'] = df.Date.apply(lambda d: (d.day-1)//7 + 1)
    # convert to string
    #df.loc[:,'wom'] = df.loc[:,'wom'].astype(str)

    # check values for week-of-month (should be between 1 and 5)
    #print(df.wom.value_counts())

    #Task 1 - remove
    #df.loc[:,'month_wom'] = df.Month + '_w' + df.wom

    # Task1 - remove
    # examples of encoding
    #df.month_wom.value_counts()[0:2]

    # Task1 - remove
    # del wom temp variable
    #del df['wom']

    # Task 1 -remove
    # what are the categorical features?
    #CATEGORICAL.append('month_wom')
    #print(CATEGORICAL)


    # Generate dummy variables (no need for bool, let's have int32 instead)
    dummy_variables = pd.get_dummies(df[CATEGORICAL], dtype='int32')

    #print(dummy_variables.info())

    # get dummies names in a list
    DUMMIES = dummy_variables.keys().to_list()
    #print(DUMMIES)

    #print(len(DUMMIES))

    # Concatenate the dummy variables with the original DataFrame
    df_with_dummies = pd.concat([df, dummy_variables], axis=1)

    # TASK1. Before 301 entries. After should be 208.
    #print(df_with_dummies[GET_NUMERICAL(df)+DUMMIES].info())

    DUMMIES_MONTH_WOM = [k for k in DUMMIES if k.startswith('month_wom')]
    # check a few records
    DUMMIES_MONTH_WOM[0:2]

    corr_month_wom_vs_is_positive_growth_30d_future = df_with_dummies[DUMMIES_MONTH_WOM+GET_TO_PREDICT(df)].corr()['is_positive_growth_30d_future']
    #print(corr_month_wom_vs_is_positive_growth_30d_future)

    # create a dataframe for an easy way to sort
    corr_month_wom_vs_is_positive_growth_30d_future_df = pd.DataFrame(corr_month_wom_vs_is_positive_growth_30d_future)

    # rename column 'is_positive_growth_5d_future' to 'corr'
    corr_month_wom_vs_is_positive_growth_30d_future_df.rename(columns={'is_positive_growth_30d_future':'corr'},inplace=True)

    corr_month_wom_vs_is_positive_growth_30d_future_df.loc[:, 'abs_corr'] = corr_month_wom_vs_is_positive_growth_30d_future_df['corr'].abs()

    #print(corr_month_wom_vs_is_positive_growth_30d_future_df.sort_values(by='abs_corr'))

    # SPLIT 1.2.4) Temporal split

    from task1_functions import temporal_split

    min_date_df = df_with_dummies.Date.min()
    max_date_df = df_with_dummies.Date.max()

    df_with_dummies = temporal_split(df_with_dummies,
                                     min_date = min_date_df,
                                     max_date = max_date_df)

    print(df_with_dummies['split'].value_counts()/len(df_with_dummies))

    # remove the "segmentation" problem (warning message on df performance after many joins and data transformations)
    save_df(df_with_dummies.copy(), 'step2.pickle')

def load_step_2() -> pd.DataFrame:
    return load_df("step2.pickle")

# MODELING ============================
#create_step_2(df)

new_df = load_step_2()
#print(new_df.info())

#print(new_df.head(1))

# time split on train/validation/test: FIXED dates of split, approx. 70%, 15%, 15% split
#print(new_df.groupby(['split'])['Date'].agg({'min','max','count'}))

# check for imbalances of growth for train/test/validation
#print(new_df.groupby(by='split')['growth_future_30d'].describe())

# what do we try to predict
#print(new_df[GET_TO_PREDICT(new_df)].head(1))

# to be used as features
#print(new_df[GET_NUMERICAL(new_df)+GET_DUMMIES(new_df)].head(1))

#1.1) Manual 'rule of thumb' predictions

#    (pred0) CCI>200 (binary, on technical indicator CCI)
#    (pred1) growth_30d>1
#    (pred2) (growth_30d>1) & (growth_snp500_30d>1)
#    (pred3) (DGS10 <= 4) & (DGS5 <= 1)
#    (pred4) (DGS10 > 4) & (FEDFUNDS <= 4.795)

# generate manual predictions
# Let's label all prediction features with prefix "pred"
new_df['pred0_manual_cci'] = (new_df.cci>200).astype(int)
new_df['pred1_manual_prev_g1'] = (new_df.growth_30d>1).astype(int)
new_df['pred2_manual_prev_g1_and_snp'] = ((new_df['growth_30d'] > 1) & (new_df['growth_snp500_30d'] > 1)).astype(int)

# new manual predictions from HA
new_df['pred3_manual_dgs10_5'] = ((new_df['DGS10'] <= 4) & (new_df['DGS5'] <= 1)).astype(int)
new_df['pred4_manual_dgs10_fedfunds'] = ((new_df['DGS10'] > 4) & (new_df['FEDFUNDS'] <= 4.795)).astype(int)

# sample of 10 observations and predictions
#print(new_df[['cci','growth_30d','growth_snp500_30d','pred0_manual_cci','pred1_manual_prev_g1','pred2_manual_prev_g1_and_snp','pred3_manual_dgs10_5','pred4_manual_dgs10_fedfunds','is_positive_growth_30d_future']].sample(10))

# List of current predictions
PREDICTIONS = [k for k in new_df.keys() if k.startswith('pred')]
#print(PREDICTIONS)

from task1_functions import get_predictions_correctness

PREDICTIONS, IS_CORRECT = get_predictions_correctness(new_df, to_predict='is_positive_growth_30d_future')

# sample of 10 predictions vs. is_correct vs. is_positive_growth_30d_future (what we're trying to predict)
#print(new_df[PREDICTIONS+IS_CORRECT+['is_positive_growth_30d_future']].sample(10))

#print(len(new_df[new_df.split=='test']))

# pred4 seems to be empty on Test - let's check why?
# it used to be some stats
#print(new_df[(new_df['gdppot_us_yoy'] >= 0.027) & (new_df['growth_wti_oil_30d'] <= 1.005)])

##### 1.2) Decision Tree Classifier

from task1_functions import clean_dataframe_from_inf_and_nan, fit_decision_tree

#1.2.2) CLF10 (Decision Tree Classifier, max_depth==10): get unique correct predictions vs. pred0_manual...pred4_manual

#    Fit(Train) on TRAIN+VALIDATION
#    Predict on ALL and Join to the original new_df
#    Get Precision on TEST

# Features to be used in predictions (incl. new dummies)
features_list = GET_NUMERICAL(new_df)+GET_DUMMIES(new_df)
# What we're trying to predict?
to_predict = 'is_positive_growth_30d_future'

train_valid_df = new_df[new_df.split.isin(['train','validation'])].copy(deep=True)
test_df = new_df[new_df.split.isin(['test'])].copy(deep=True)

# ONLY numerical Separate features and target variable for training and testing sets
X_train_valid = train_valid_df[features_list+[to_predict]]
X_test = test_df[features_list+[to_predict]]

# this to be used for predictions and join to the original dataframe new_df
X_all =  new_df[features_list+[to_predict]].copy(deep=True)

#print(f'length: X_train_valid {X_train_valid.shape},  X_test {X_test.shape}, all combined: X_all {X_all.shape}')

# Clean from +-inf and NaNs:

X_train_valid = clean_dataframe_from_inf_and_nan(X_train_valid)
# X_test = clean_dataframe_from_inf_and_nan(X_test) # won't use
X_all = clean_dataframe_from_inf_and_nan(X_all)

y_train_valid = X_train_valid[to_predict]
# y_test = X_test[to_predict] # won't use
y_all =  X_all[to_predict]

# remove y_train, y_test from X_ dataframes
del X_train_valid[to_predict]
del X_test[to_predict]
del X_all[to_predict]

# Create TREE
#start_time = time.time()
#clf_10, train_columns = fit_decision_tree(X=X_train_valid,
#                           y=y_train_valid,
#                           max_depth=10)
#print(f"Execution time: {time.time() - start_time:.4f} seconds")
#save_tree(clf_10, "task1_clf10.joblib")

# Load TREE
clf_10 = load_tree("task1_clf10.joblib")
train_columns = X_train_valid.columns
#

#print(X_train_valid.shape)
#print(X_all.shape)
#print(y_all.shape)

# predict on a full dataset
y_pred_all = clf_10.predict(X_all)

# defining a new prediction vector is easy now, as the dimensions will match
new_df['pred5_clf_10'] = y_pred_all

# new prediction is added --> need to recalculate the correctness
PREDICTIONS, IS_CORRECT = get_predictions_correctness(new_df, to_predict='is_positive_growth_30d_future')

#print(IS_CORRECT)

# define a new column that find the cases when only pred5 is correct
new_df['only_pred5_is_correct'] = (new_df.is_correct_pred5==new_df.is_positive_growth_30d_future) & \
                         (new_df.is_positive_growth_30d_future == 1) & \
                         (new_df.is_correct_pred0 == 0) & \
                         (new_df.is_correct_pred1 == 0) & \
                         (new_df.is_correct_pred2 == 0) & \
                         (new_df.is_correct_pred3 == 0) & \
                         (new_df.is_correct_pred4 == 0)

# need it to be integer and not bool
new_df['only_pred5_is_correct'] = new_df['only_pred5_is_correct'].astype(int)

# how many times only pred5 is correct in the TEST set?
#print(new_df[new_df.split=='test']['only_pred5_is_correct'].sum())

# let's look at the record
filter_unique_pred_5 = (new_df.split=='test') & (new_df.only_pred5_is_correct==1)

# sample with only Pred5 correct
#print(new_df[filter_unique_pred_5].sample(10))

# let's visually check that all predictions are 0, but pred5==1 (sample 10)
#print(new_df[filter_unique_pred_5][PREDICTIONS+IS_CORRECT+['is_positive_growth_30d_future']].sample(10))

save_df(new_df, "task_1_result.pickle")
