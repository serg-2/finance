import pandas as pd

from task1_functions import clean_dataframe_from_inf_and_nan, get_predictions_correctness
from task_snippet_0 import load_tree, save_tree, GET_NUMERICAL,GET_DUMMIES,load_df,save_df
from joblib import dump

def load_task1_result() -> pd.DataFrame:
    return load_df("task_1_result.pickle")

new_df = load_task1_result()

### 1.2.3 [Code Snippet 2] Hyperparams tuning for a Decision Tree Classifier

# Features to be used in predictions (incl. new dummies)
features_list = GET_NUMERICAL(new_df)+GET_DUMMIES(new_df)
# What we're trying to predict?
to_predict = 'is_positive_growth_30d_future'

train_df = new_df[new_df.split.isin(['train'])].copy(deep=True)
valid_df = new_df[new_df.split.isin(['validation'])].copy(deep=True)
train_valid_df = new_df[new_df.split.isin(['train','validation'])].copy(deep=True)

test_df =  new_df[new_df.split.isin(['test'])].copy(deep=True)

# ONLY numerical Separate features and target variable for training and testing sets
X_train = train_df[features_list+[to_predict]]
X_valid = valid_df[features_list+[to_predict]]

X_train_valid = train_valid_df[features_list+[to_predict]]

X_test = test_df[features_list+[to_predict]]

# this to be used for predictions and join to the original dataframe new_df
X_all =  new_df[features_list+[to_predict]].copy(deep=True)

#print(f'length: X_train {X_train.shape},  X_validation {X_valid.shape}, X_test {X_test.shape}, X_train_valid = {X_train_valid.shape},  all combined: X_all {X_all.shape}')

# Clean from +-inf and NaNs:
X_train = clean_dataframe_from_inf_and_nan(X_train)
X_valid = clean_dataframe_from_inf_and_nan(X_valid)
X_train_valid = clean_dataframe_from_inf_and_nan(X_train_valid)
X_test = clean_dataframe_from_inf_and_nan(X_test)
X_all = clean_dataframe_from_inf_and_nan(X_all)

y_train = X_train[to_predict]
y_valid = X_valid[to_predict]
y_train_valid = X_train_valid[to_predict]
y_test = X_test[to_predict]
y_all =  X_all[to_predict]

# remove y_train, y_test from X_ dataframes
del X_train[to_predict]
del X_valid[to_predict]
del X_train_valid[to_predict]
del X_test[to_predict]
del X_all[to_predict]

# visualisation: decision tree for a few levels (max_depth variable)
#from sklearn.tree import plot_tree
#import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/20156951/how-do-i-find-which-attributes-my-tree-splits-on-when-using-scikit-learn
#from sklearn.tree import export_text

# (8min runtime) UNCOMMENT TO RUN IT AGAIN
# %%time
# # hyper params tuning for a Decision Tree

# precision_by_depth = {}
# best_precision = 0
# best_depth = 0

# for depth in range(1,21):
#   print(f'Working with a tree of a max depth= {depth}')
#   # fitting the tree on X_train, y_train
#   clf,train_columns = fit_decision_tree(X=X_train_valid,
#                            y=y_train_valid,
#                            max_depth=depth) #applying custom hyperparam
#   # getting the predictions for TEST and accuracy score
#   y_pred_valid = clf.predict(X_valid)
#   precision_valid = precision_score(y_valid, y_pred_valid)
#   y_pred_test = clf.predict(X_test)
#   precision_test = precision_score(y_test, y_pred_test)
#   print(f'  Precision on test is {precision_test}, (precision on valid is {precision_valid} - tend to overfit)')
#   # saving to the dict
#   precision_by_depth[depth] = round(precision_test,4)
#   # updating the best precision
#   if precision_test >= best_precision:
#     best_precision = round(precision_test,4)
#     best_depth = depth
#   # plot tree - long
#   # plt.figure(figsize=(20,10))  # Set the size of the figure
#   # plot_tree(clf,
#   #           filled=True,
#   #           feature_names=train_columns,
#   #           class_names=['Negative', 'Positive'],
#   #           max_depth=2)
#   # plt.show()
#   # plot tree - short
#   tree_rules = export_text(clf, feature_names=list(X_train), max_depth=3)
#   print(tree_rules)
#   print('------------------------------')

# print(f'All precisions by depth: {precision_by_depth}')
# print(f'The best precision is {best_precision} and the best depth is {best_depth} ')


# Results of Hyper parameters tuning for a Decision Tree
# print(precision_by_depth)

# LOAD
precision_by_depth = {1: 0.5466, 2: 0.5511, 3: 0.5511, 4: 0.5511, 5: 0.6278, 6: 0.5691, 7: 0.5945, 8: 0.5891, 9: 0.5912, 10: 0.5888, 11: 0.5916, 12: 0.5855, 13: 0.5822, 14: 0.592, 15: 0.5833, 16: 0.5898, 17: 0.586, 18: 0.5861, 19: 0.5869, 20: 0.5773}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(list(precision_by_depth.items()), columns=['max_depth', 'precision_score'])
df.loc[:,'precision_score'] = df.precision_score*100.0 # need for % visualisation

#from task1_figures import show_fig
#show_fig(df)

#TREE
#clf_5, train_columns = fit_decision_tree(X=X_train_valid,
#                           y=y_train_valid,
#                           max_depth=5)
#save_tree(clf_5, "task1_clf5.joblib")
clf_5 = load_tree("task1_clf5.joblib")
train_columns = X_train_valid.columns

# predict on a full dataset
y_pred_all = clf_5.predict(X_all)

# defining a new prediction vector is easy now, as the dimensions will match
new_df['pred6_clf_5'] = y_pred_all

# MANUAL SECOND BEST, need some complexity
# found earlier in HyperParams Tuning
best_depth = 14
best_precision = precision_by_depth[best_depth]

#print(f'Best precision and depth = {best_depth}, precision (on test)={best_precision}')

#TREE
#clf_best, train_columns = fit_decision_tree(X=X_train_valid,
#                           y=y_train_valid,
#                           max_depth=best_depth)
#save_tree(clf_best, "task1_clfbest.joblib")

clf_best=load_tree("task1_clfbest.joblib")
train_columns = X_train_valid.columns

# For a DecisionTreeClassifier in scikit-learn, the concept of trainable parameters differs from that of neural networks.
# In decision trees, the parameters are the structure of the tree itself (nodes and splits) rather than weights.
# However, you can still get a sense of the model's complexity by looking at the number of nodes and leaves.

# Here's how you can get this information for your trained DecisionTreeClassifier (referred to as clf_best):

# Get the number of nodes and leaves in the tree
n_nodes = clf_best.tree_.node_count
n_leaves = clf_best.get_n_leaves()

#print(f"Number of nodes: {n_nodes}")
#print(f"Number of leaves: {n_leaves}")
#print(clf_best)

# predict on a full dataset
y_pred_clf_best = clf_best.predict(X_all)

# defining a new prediction vector is easy now, as the dimensions will match
new_df['pred7_clf_second_best'] = y_pred_clf_best

# new prediction is added --> need to recalculate the correctness
PREDICTIONS, IS_CORRECT = get_predictions_correctness(new_df, to_predict='is_positive_growth_30d_future')

dump(PREDICTIONS, "task1_predictions.joblib")

### 1.2.4) Two ways of visualisation

# train a simple tree
#clf_2,train_columns = fit_decision_tree(X=X_train_valid,
#                           y=y_train_valid,
#                           max_depth=2)
#save_tree(clf_2, "task1_clf2.joblib")
clf_2 = load_tree("task1_clf2.joblib")
train_columns = X_train_valid.columns

#from task1_figures import show_tree
#show_tree(clf_2, train_columns)


from sklearn.tree import export_text

tree_rules = export_text(clf_2, feature_names=list(X_train), max_depth=1)
print(tree_rules)

save_df(X_train_valid, "task_1_result_X.pickle")
save_df(y_train_valid, "task_1_result_Y.pickle")

save_df(X_valid, "task_1_valid_X.pickle")
save_df(X_test, "task_1_test_X.pickle")

save_df(X_train, "task_1_train_X.pickle")

save_df(y_test, "task_1_test_Y.pickle")

save_df(X_all, "task_1_all_X.pickle")

save_df(new_df, "task1_step2.pickle")
