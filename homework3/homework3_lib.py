import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.impute import SimpleImputer
from typing import Tuple

def get_numerical(df_full: pd.DataFrame) -> list:
    GROWTH = [g for g in df_full.keys() if (g.find('growth_')==0)&(g.find('future')<0)]
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

    TECHNICAL_PATTERNS = [g for g in df_full.keys() if g.find('cdl')>=0]
    print(f'Technical patterns count = {len(TECHNICAL_PATTERNS)}, examples = {TECHNICAL_PATTERNS[0:5]}')

    MACRO = ['gdppot_us_yoy', 'gdppot_us_qoq', 'cpi_core_yoy', 'cpi_core_mom', 'FEDFUNDS',
     'DGS1', 'DGS5', 'DGS10']
    
    # manually defined features
    CUSTOM_NUMERICAL = ['SMA10', 'SMA20', 'growing_moving_average', 'high_minus_low_relative','volatility', 'ln_volume']

    NUMERICAL = GROWTH + TECHNICAL_INDICATORS + TECHNICAL_PATTERNS + CUSTOM_NUMERICAL + MACRO
    return NUMERICAL

def get_dummies(new_df: pd.DataFrame) -> list:
    dummy_variables = pd.get_dummies(new_df[['Month', 'Weekday', 'Ticker', 'ticker_type']], dtype='int32')
    # get dummies names in a list
    return dummy_variables.keys().to_list()

def get_dummies2(new_df: pd.DataFrame) -> list:
    dummy_variables = pd.get_dummies(new_df[['Month', 'Weekday', 'Ticker', 'ticker_type', 'month_wom']], dtype='int32')
    # get dummies names in a list
    return dummy_variables.keys().to_list()

def remove_outliers_percentile(X, lower_percentile=1, upper_percentile=99):
    """
    Remove outliers from the input array based on percentiles.

    Parameters:
    - X: Input array (NumPy array or array-like)
    - lower_percentile: Lower percentile threshold (float, default=1)
    - upper_percentile: Upper percentile threshold (float, default=99)

    Returns:
    - Array with outliers removed
    """
    lower_bound = np.percentile(X, lower_percentile, axis=0)
    upper_bound = np.percentile(X, upper_percentile, axis=0)
    mask = np.logical_and(np.all(X >= lower_bound, axis=1), np.all(X <= upper_bound, axis=1))
    return X[mask]

# estimation/fit function (using dataframe of features X and what to predict y) --> optimising total accuracy
# max_depth is hyperParameter
def fit_decision_tree(X, y, max_depth):
# Initialize the Decision Tree Classifier
  clf = DecisionTreeClassifier(max_depth = max_depth, random_state = 42)

  # Fit the classifier to the training data
  clf.fit(X, y)
  return clf, X.columns

def predict_decision_tree(clf:DecisionTreeClassifier, df_X:pd.DataFrame, y_true: pd.Series) -> Tuple[pd.DataFrame, float]:
  # Predict the target variable on the test data
  y_pred = clf.predict(df_X)

  #max_depth = clf.tree_.max_depth
  # Print the maximum depth
  #print("Maximum depth of the decision tree:", max_depth)

  # Calculate the accuracy/precision of the model
  #accuracy = accuracy_score(y_test, y_pred)
  #precision = precision_score(y_test, y_pred)
  #print(f'Accuracy ={accuracy}, precision = {precision}')

  # resulting df
  result_df = pd.concat([df_X, y_true, pd.Series(y_pred, index=df_X.index, name='pred_')], axis=1)

  return result_df, precision_score(y_true, y_pred)
