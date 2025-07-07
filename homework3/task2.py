#!/usr/bin/env python

import pandas as pd
import pickle
from homework3_lib import get_numerical, get_dummies, get_dummies2

def save_df(df: pd.DataFrame, filename: str):
    with open(filename, 'wb') as file:
        pickle.dump(df, file)

def load_df(filename: str) -> pd.DataFrame:
    with open(filename, 'rb') as file:
        map_data = pickle.load(file)
    return map_data

def temporal_split(df, min_date, max_date, train_prop=0.7, val_prop=0.15, test_prop=0.15):
    """
    Splits a DataFrame into three buckets based on the temporal order of the 'Date' column.

    Args:
        df (DataFrame): The DataFrame to split.
        min_date (str or Timestamp): Minimum date in the DataFrame.
        max_date (str or Timestamp): Maximum date in the DataFrame.
        train_prop (float): Proportion of data for training set (default: 0.6).
        val_prop (float): Proportion of data for validation set (default: 0.2).
        test_prop (float): Proportion of data for test set (default: 0.2).

    Returns:
        DataFrame: The input DataFrame with a new column 'split' indicating the split for each row.
    """
    # Define the date intervals
    train_end = min_date + pd.Timedelta(days=(max_date - min_date).days * train_prop)
    val_end = train_end + pd.Timedelta(days=(max_date - min_date).days * val_prop)

    # Assign split labels based on date ranges
    split_labels = []
    for date in df['Date']:
        if date <= train_end:
            split_labels.append('train')
        elif date <= val_end:
            split_labels.append('validation')
        else:
            split_labels.append('test')

    # Add 'split' column to the DataFrame
    df['split'] = split_labels

    return df

#1 Load data from file
def load_hand_rules() -> pd.DataFrame:
    df_with_dummies = load_df('dummies.pickle')
    #print(df_with_dummies.tail(1))

    df_with_dummies = temporal_split(df_with_dummies,
                                     min_date = df_with_dummies.Date.min(),
                                     max_date = df_with_dummies.Date.max())

    print(df_with_dummies['split'].value_counts()/len(df_with_dummies))
    new_df = df_with_dummies.copy()

    print(new_df.head(1))
    print()
    print(new_df.groupby(by='split')['growth_future_30d'].describe())
    print()
    print(new_df.groupby(['split'])['Date'].agg({'min','max','count'}))

    # what we try to predict
    TO_PREDICT = [g for g in new_df.keys() if (g.find('future')>=0)]
    print("TO PREDICT: ",new_df[TO_PREDICT].head(1))

    # to be used as features

    NUMERICAL = get_numerical(new_df)
    print("NUMERICAL", NUMERICAL)

    # get dummies names in a list
    DUMMIES = get_dummies2(new_df)

    print(new_df[NUMERICAL+DUMMIES].head(1))

    #Hand rules
    new_df['pred0_manual_cci'] = (new_df.cci>200).astype(int)
    new_df['pred1_manual_prev_g1'] = (new_df.growth_30d>1).astype(int)
    new_df['pred2_manual_prev_g1_and_snp'] = ((new_df['growth_30d'] > 1) & (new_df['growth_snp500_30d'] > 1)).astype(int)

    # (DGS10 <= 4) & (DGS5 <= 1)
    new_df['pred3_manual_dgs10_5'] = ((new_df['DGS10'] <= 4) & (new_df['DGS5'] <= 1)).astype(int)
    # (DGS10 > 4) & (FEDFUNDS <= 4.795)
    new_df['pred4_manual_dgs10_fedfunds'] = ((new_df['DGS10'] > 4) & (new_df['FEDFUNDS'] <= 4.795)).astype(int)
    return new_df

# Make
save_df(load_hand_rules(), "step1.pickle")

# Load
new_df = load_df("step1.pickle")

#print(new_df[['growth_30d','is_positive_growth_30d_future', 'pred3_manual_dgs10_5', 'pred4_manual_dgs10_fedfunds']].sample(10))

new_df['is_correct_pred0'] = (new_df.pred0_manual_cci == new_df.is_positive_growth_30d_future)
new_df['is_correct_pred1'] = (new_df.pred1_manual_prev_g1 == new_df.is_positive_growth_30d_future)
new_df['is_correct_pred2'] = (new_df.pred2_manual_prev_g1_and_snp == new_df.is_positive_growth_30d_future)

new_df['is_correct_pred3'] = (new_df.pred3_manual_dgs10_5 == new_df.is_positive_growth_30d_future)
new_df['is_correct_pred4'] = (new_df.pred4_manual_dgs10_fedfunds == new_df.is_positive_growth_30d_future)

#print(new_df[['pred3_manual_dgs10_5','is_positive_growth_30d_future','is_correct_pred3']])
#print(new_df[['pred4_manual_dgs10_fedfunds','is_positive_growth_30d_future','is_correct_pred4']])

#Filter
filter_pred3 = (new_df.split=='test') & (new_df.pred3_manual_dgs10_5==1)
filter_pred4 = (new_df.split=='test') & (new_df.pred4_manual_dgs10_fedfunds==1)

#print(new_df[filter_pred3].is_correct_pred3.value_counts())
print(new_df[filter_pred3].is_correct_pred3.value_counts() / len(new_df[filter_pred3]))
#print(new_df[filter_pred4].is_correct_pred4.value_counts())
print(new_df[filter_pred4].is_correct_pred4.value_counts() / len(new_df[filter_pred4]))

save_df(new_df, 'step2.pickle')
