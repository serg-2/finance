import pandas as pd
import plotly.express as px
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier


# Create the bar chart using Plotly Express
def show_fig(df: pd.DataFrame):
    fig = px.bar(df,
                 x='max_depth',
                 y='precision_score',
                #  title='Precision Score vs. Max Depth for a Decision Tree',
                 labels={'max_depth': 'Max Depth', 'precision_score': 'Precision Score'},
                 range_y=[54, 65],
                 text='precision_score')

    # Update the text format to display as percentages
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')

    # Center the title
    fig.update_layout(title={'text': 'Precision Score vs. Max Depth for a Decision Tree', 'x': 0.5, 'xanchor': 'center'})

    # Show the figure
    fig.show()

# Visualisation: decision tree for a few levels (max_depth variable)
#def show_tree(clf_2: DecisionTreeClassifier, train_columns: pd.Index[str]):
def show_tree(clf_2: DecisionTreeClassifier, train_columns):
    print("******************")
    print(type(train_columns))
    print("******************")

    # Assuming clf_2 is your trained DecisionTreeClassifier
    plt.figure(figsize=(12,10))  # Set the size of the figure
    plot_tree(clf_2,
          filled=True,
          feature_names=train_columns,
          class_names=['Negative', 'Positive'],
          max_depth=2)
    plt.show()

# Create the bar chart using Plotly Express
def show_fig_snippet3(df: pd.DataFrame):
    # Create line plot using Plotly Express
    fig = px.line(df, x='max_depth', y='precision_score', color='n_estimators',
              labels={'max_depth': 'Max Depth', 'precision_score': 'Precision Score', 'n_estimators': 'Number of Estimators'},
              title='Random Forest Models: Precision Score vs. Max Depth for Different Number of Estimators')

    # Adjust x-axis range
    fig.update_xaxes(range=[5, 20])

    # Show the figure
    fig.show()

def show_fig_1_3(df: pd.DataFrame):
    # Create the bar chart
    fig = px.bar(df,
                 x='Combination',
                 y='Precision',
                 text='Precision'
                 )

    # Customize the layout for better readability
    fig.update_layout(
        xaxis_title='Hyperparams combinations of <C, Max Iterations>',
        yaxis_title='Precision Score',
        xaxis_tickangle=-45,
        title={
            'text': 'Precision Scores for Various Logistic Regression Hyperparameter Combinations',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )


    # Update the text position
    fig.update_traces(texttemplate='%{text:.2f}%',
                      textposition='inside',
                      textfont_color='white')

    # Show the figure
    fig.show()

def show_plt_snippet4(best_history_nn:dict):
    # Learning visualisation for the Deep Neural Network (DNN)
    # The model is not actually training, as the precision and accuracy score are not improving on TRAIN/TEST with more Epochs

    # Plotting accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(best_history_nn['precision_1'], label='Training Precision')
    plt.plot(best_history_nn['val_precision_1'], label='Test Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Training vs. Test Precision')
    plt.legend()
    plt.grid(True)
    plt.show()
