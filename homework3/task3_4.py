#!/usr/bin/env python

import numpy as np
import pandas as pd
import pickle
from homework3_lib import get_numerical, get_dummies, remove_outliers_percentile, fit_decision_tree, predict_decision_tree

def save_df(df: pd.DataFrame, filename: str):
    with open(filename, 'wb') as file:
        pickle.dump(df, file)

def load_df(filename: str) -> pd.DataFrame:
    with open(filename, 'rb') as file:
        map_data = pickle.load(file)
    return map_data

# Load
new_df = load_df("step2.pickle")

features_list = get_numerical(new_df) + get_dummies(new_df)
to_predict = 'is_positive_growth_30d_future'

train_df = new_df[new_df.split.isin(['train','validation'])].copy(deep=True)
test_df = new_df[new_df.split.isin(['test'])].copy(deep=True)

#all_df = new_df.copy(deep=True)
all_df = new_df[new_df.split.isin(['train','validation', 'test'])].copy(deep=True)

# ONLY numerical Separate features and target variable for training and testing sets
# need Date and Ticker later when merging predictions to the dataset
X_train = train_df[features_list+[to_predict,'Date','Ticker']]
X_test = test_df[features_list+[to_predict,'Date','Ticker']]
X_all = all_df[features_list+[to_predict,'Date','Ticker']]

# Disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_all.replace([np.inf, -np.inf], np.nan, inplace=True)

# Need to fill NaNs somehow
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)
X_all.fillna(0, inplace=True)

X_train_imputed = X_train # we won't use outliers removal to save more data to train: remove_outliers_percentile(X_train)
X_test_imputed = X_test # we won't use outliers removal to save more data to test: remove_outliers_percentile(X_test)
X_all_imputed = X_all

y_train = X_train_imputed[to_predict]
y_test = X_test_imputed[to_predict]
y_all = X_all_imputed[to_predict]

# remove y_train, y_test from X_ dataframes
del X_train_imputed[to_predict]
del X_test_imputed[to_predict]
del X_all_imputed[to_predict]

# INPUTS: ========================================================================
# X_train_imputed : CLEAN dataFrame with only numerical features (train+validation periods)
# X_test_imputed : CLEAN dataFrame with only numerical features (test periods)
# X_all_imputed : CLEAN dataFrame with only numerical features (all periods)

# y_train : true values for the train period
# y_test  : true values for the test period
# y_all  : true values for the all period

results = []

for i in range(1, 21):
    # drop 2 columns before fitting the tree, but we need those columns later for joins
    clf_10, train_columns = fit_decision_tree(X=X_train_imputed.drop(['Date','Ticker'],axis=1),
                               y=y_train,
                               max_depth=i)

    # Predict on ALL!
    ###pred10 = predict_decision_tree(clf_10, X_test_imputed.drop(['Date','Ticker'],axis=1), y_test)
    #pred10 = predict_decision_tree(clf_10, X_all_imputed.drop(['Date','Ticker'],axis=1), y_all)

    #print(pred10)
    #print(pred10.pred_.value_counts())

    # define a new DF with the SAME index (used for joins)
    #pred10_df = pred10[['pred_']].rename(columns={'pred_': 'pred5_clf_10'})
    #print(pred10_df.head(1))

    # Store these predictions in a new column named pred5_clf_10 within your main dataframe.
    #df_result = pd.concat([new_df, pred10_df], axis=1)

    # For all
    #df_result = new_df.copy(deep=True)
    # For test
    df_result = new_df[new_df.split.isin(['test'])].copy(deep=True)
    
    res, precision_score = predict_decision_tree(
        clf_10, 
        #X_all_imputed.drop(['Date','Ticker'],axis=1), 
        #y_all
        X_test_imputed.drop(['Date','Ticker'],axis=1), 
        y_test
        )
    df_result["pred5_clf_10"] = res[['pred_']]

    print(df_result.head(5))

    df_result['is_correct_mainpred'] = (
        ((df_result.is_correct_pred0 == 0) & 
         (df_result.is_correct_pred1 == 0) & 
         (df_result.is_correct_pred2 == 0) & 
         (df_result.is_correct_pred3 == 0) & 
         (df_result.is_correct_pred4 == 0) & 
         (df_result.pred5_clf_10 == 1))
        .astype(int)
    )

    print(df_result.is_correct_mainpred.value_counts())

    answer_df = df_result[df_result.split.isin(['test'])].copy(deep=True)

    print(answer_df.is_correct_mainpred.value_counts())

    # 42 (10) - 3817
    # 41 (10) - 3767
    # 40 (10) - 3840
    # 42 (20) - 3480

    # 42 (10) validation - 3653

    tuple2 = (i, precision_score)
    results.append(tuple2)

print(results)
