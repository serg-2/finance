from task1 import load_df
from task1_functions import get_predictions_correctness
from joblib import dump

#2.3) Agg. stats on ALL predictions

new_df = load_df("task1_snippet6.pickle")

# let's review the Predictions:
PREDICTIONS, IS_CORRECT = get_predictions_correctness(new_df, to_predict='is_positive_growth_30d_future')

# check approx. periods : Train is 2000-01...2017-01, Valid is 2017-01..2020-09, Test is 2020-09..2024-05
#print(new_df.groupby('split').Date.agg(['min','max']))

#print(PREDICTIONS)

PREDICTIONS_ON_MODELS = [p for p in PREDICTIONS if int(p.split('_')[0].replace('pred', ''))>=5]
#print(PREDICTIONS_ON_MODELS)

IS_CORRECT_ON_MODELS = [p for p in IS_CORRECT if int(p.replace('is_correct_pred', ''))>=5]
#print(IS_CORRECT_ON_MODELS)

# predictions on models
# pred10_rf_best_rule_60: ONLY 2% of TEST cases predicted with high confidence of growth
#print(new_df.groupby('split')[PREDICTIONS_ON_MODELS].agg(['count','sum','mean']).T)

# 10 predictions stats (check TEST set)
print(new_df.groupby('split')[PREDICTIONS].agg(['count','sum','mean']).T)

dump(PREDICTIONS, "snippet6_predictions.joblib")