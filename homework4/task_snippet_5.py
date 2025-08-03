import pandas as pd
from task_snippet_0 import load_df, load_tree


# Preload
X_test_load = load_df("task_1_test_X.pickle")
X_test = X_test_load.copy(deep=True)
y_test_load = load_df("task_1_test_Y.pickle")
y_test = y_test_load.copy(deep=True)


clf_best=load_tree("task1_clfbest.joblib")


#2) [Code Snippet 5] Different Decision rules to improve the Precision (varying Threshold)

#    best model1 (clf_best): Decision Tree (max_depth=15)
#    best model2 (rf_best): Random Forest (n_estimators=200, max_depth=17)


#2.1. Predicting probabilities (predict_proba), getting the distribution for probabilities, and new decision rules

# predicting probability instead of a label
y_pred_test = clf_best.predict_proba(X_test)

# y_pred_test = rf_best.predict_proba(X_test)

y_pred_test_class1 = [k[1] for k in y_pred_test] # k[1] is the second element in the list of Class predictions

# example prediction
#print(y_pred_test)

y_pred_test_class1_df = pd.DataFrame(y_pred_test_class1, columns=['Class1_probability'])

# sample of predictions
#print(y_pred_test_class1_df.sample(10))

# Mean prediction is 0.3, median is 0.0, 75% quantile is 0.9
print(y_pred_test_class1_df.describe().T)

# Unconditional probability of a positive growth is 55.5%
print(y_test.sum()/y_test.count())

# FIGURE
#sns.histplot(y_pred_test_class1)
# Add a title
#plt.title(f'The distribution of predictions for the current second best model (Decision Tree with max_depth={clf_best.get_depth()})')
# Show the plot
#plt.show()

from task1_functions import tpr_fpr_dataframe

df_scores = tpr_fpr_dataframe(y_test,
                              y_pred_test_class1,
                              only_even=True)

#print(df_scores)

print(df_scores[(df_scores.threshold>=0.6) & (df_scores.threshold<=0.92)])

# PLOT
# Try to find high Precision score points

#df_scores.plot.line(x='threshold',
#                    y=['precision','recall', 'f1_score'],
#                    title = 'Precision vs. Recall for the Best Model (Decision Tree with max_depth=15)')
