import numpy as np
import pandas as pd
from joblib import load

from task1 import load_df, load_tree, save_df
from task_snippet_3_0 import load_tree_random

#2.2) [Code Snippet 6] Defining new columns with Predictions in new_df: pred7..pred10

#    pred7 and pred8 are 2 decision rules for the Decision Tree (best model with max_depth=15)
#    pred9 and pred10 are 2 decision rules for the Random Forest (second best model)


# Preload

PREDICTIONS = load("task1_predictions.joblib")
X_all_load = load_df("task_1_all_X.pickle")
X_all = X_all_load.copy(deep=True)

clf_best=load_tree("task1_clfbest.joblib")

new_df_load = load_df("task1_step2.pickle")
new_df = new_df_load.copy(deep=True)

#print(PREDICTIONS)

# adding Decision Tree predictors (clf_best) to the dataset for 2 new rules: Threshold = 0.66 and 0.78


y_pred_all = clf_best.predict_proba(X_all)
y_pred_all_class1 = [k[1] for k in y_pred_all] #list of predictions for class "1"
y_pred_all_class1_array = np.array(y_pred_all_class1) # (Numpy Array) np.array of predictions for class "1" , converted from a list

# defining a new prediction vector is easy now, as the dimensions will match
new_df['proba_pred8'] = y_pred_all_class1_array
new_df['pred8_clf_second_best_rule_84'] = (y_pred_all_class1_array >= 0.84).astype(int)

new_df['proba_pred9'] = y_pred_all_class1_array
new_df['pred9_clf_second_best_rule_92'] = (y_pred_all_class1_array >= 0.92).astype(int)

rf_best = load_tree_random("step3_rf_best.joblib")
# adding Random Forest predictors (rf_best)
#print(rf_best)

# make predictions of probabilities using the Random Forest model (rf_best)
y_pred_all = rf_best.predict_proba(X_all)
y_pred_all_class1 = [k[1] for k in y_pred_all] #list of predictions for class "1"
y_pred_all_class1_array = np.array(y_pred_all_class1) # (Numpy Array) np.array of predictions for class "1" , converted from a list

# PLOT
# PREDICTIONS ON A FULL DATASET - more smooth dataset - good sign
#sns.histplot(y_pred_all_class1)
# Add a title
#plt.title('The distribution of predictions for the current best model for Random Forest)')
# Show the plot
#plt.show()

# adding Random Forest predictors (rf_best) to the dataset for 2 new rules: Threshold = 0.60 and 0.70
# defining a new prediction vector is easy now, as the dimensions will match
new_df['proba_pred10'] = y_pred_all_class1_array
new_df['pred10_rf_best_rule_55'] = (y_pred_all_class1_array >= 0.55).astype(int)

new_df['proba_pred11'] = y_pred_all_class1_array
new_df['pred11_rf_best_rule_65'] = (y_pred_all_class1_array >= 0.65).astype(int)

# Many positive predictions
#FIG
new_df[(new_df.split=='test')&(new_df.pred10_rf_best_rule_55==1)].Date.hist()

# When did it predict to trade for the "rare" positive prediction pred10?
# FIG
#new_df[(new_df.split=='test')&(new_df.pred11_rf_best_rule_65==1)].Date.hist()

# sample of rare predictions with high threshold
#print(new_df[(new_df.split=='test')&(new_df.pred11_rf_best_rule_65==1)].sort_values(by='Date').sample(10))

# List of ALL current predictions
PREDICTIONS = [k for k in new_df.keys() if k.startswith('pred')]
#print(PREDICTIONS)

# Pred 10: How many positive prediction per day (out of 33 stocks possible)
pred10_daily_positive_count = pd.DataFrame(new_df[(new_df.split=='test')&(new_df.pred11_rf_best_rule_65==1)].groupby('Date')['pred11_rf_best_rule_65'].count())

# Pred 9: How many positive prediction per day (out of 33 stocks possible)
pred9_daily_positive_count = pd.DataFrame(new_df[(new_df.split=='test')&(new_df.pred10_rf_best_rule_55==1)].groupby('Date')['pred10_rf_best_rule_55'].count())

# Unique trading days on Test (4 years)
#print(new_df[(new_df.split=='test')].Date.nunique())

#print(pred10_daily_positive_count)

#FIG
#pred10_daily_positive_count.hist()

# 75% cases we have not more than 6 bets of $100
#print(pred10_daily_positive_count.describe().T)

#FIG
#pred9_daily_positive_count.hist()

#print(pred9_daily_positive_count.describe().T)

save_df(new_df,"task1_snippet6.pickle")
