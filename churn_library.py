"""
Module:
Name: Jayden Cho
Date: May 2022
"""

# import libraries
from ast import Constant
import constants
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report, roc_curve, roc_auc_score
import os
import logging
import shap
import joblib
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(df_path):
    '''
    Returns dataframe for the csv found at pth

    Input:
            pth: a path to the csv
    Output:
            df: pandas dataframe
    '''
    df = pd.read_csv(df_path)

    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

    return df


def plot_df_helper(df_to_plot, plot_name, file_path):
    '''
    Accepts dataframe to plot and save the dataframe as a figure

    Input:
            df_to_plot: (DataFrame) dataframe to plot
            plot_name: (str) name to be used as file name
            file_path: path to save figure
    Output:
            None
    '''
    fig, ax = plt.subplots(figsize=(12, 9))

    ax.set_frame_on(False)  # No frame
    ax.xaxis.set_visible(False)  # Hide the x axis
    ax.yaxis.set_visible(False)  # Hide the y axis

    table = pd.plotting.table(ax, df_to_plot)  # where df is your data frame

    file_path += plot_name + '.png'
    plt.savefig(file_path)


def plt_plot_helper(df_to_plot, plot_type, col_name, file_path):
    '''
    Accepts dataframe to plot and save the plot of specified type as a figure

    Input:
            df_to_plot: (DataFrame) dataframe to plot
            plot_type: (str) type of plot in plt.plot()
            col_name: (str) column name for univariate eda plot
            file_path: path to save figure
    Output:
            None
    '''
    plt.figure(figsize=(20, 10))
    df_to_plot.plot(kind=plot_type)

    file_path += col_name + '.png'
    plt.savefig(file_path)


def perform_eda(df):
    '''
    Perform eda on df and save figures to images folder
    Input:
            df: pandas dataframe

    Output:
            None
    '''

    plot_df_helper(df.describe(), 'df_description', constants.EDA_OUT_PATH)
    plt_plot_helper(df['Churn'], 'hist', 'Churn', constants.EDA_OUT_PATH)
    plt_plot_helper(df['Customer_Age'], 'hist', 'Customer_Age', constants.EDA_OUT_PATH)
    plt_plot_helper(df['Marital_Status'].value_counts(
        'normalize'), 'bar', 'Marital_Status', constants.EDA_OUT_PATH)


def encoder_helper(df, category_lst, response):
    '''
    Helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    Input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name 
                [optional argument that could be used for naming variables or index y column]

    Output:
            df: pandas dataframe with new columns for
    '''
    for category in category_lst:
        new_col_name = category + '_' + response
        df[new_col_name] = df.groupby(category)[response].transform(lambda x: x.mean())

    return df


def perform_feature_engineering(df, category_lst, response):
    '''
    Input:
            df: pandas dataframe
            response: string of response name 
                [optional argument that could be used for naming variables or index y column]

    output:
            X_train: X training data
            X_test: X testing data
            y_train: y training data
            y_test: y testing data
    '''
    encoded_df = encoder_helper(df, category_lst, response)

    X = pd.DataFrame()
    X[constants.KEEP_COLS] = encoded_df[constants.KEEP_COLS]
    y = df[response]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    Produces classification report for training and testing results and stores report as image
    in images folder
    Input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    Output:
             None
    '''
    rf_test_report = pd.DataFrame.from_dict(
        classification_report(
            y_test,
            y_test_preds_rf,
            output_dict=True))
    plot_df_helper(rf_test_report, 'rf_test_report', constants.RESULTS_OUT_PATH)

    rf_train_report = pd.DataFrame.from_dict(
        classification_report(
            y_train,
            y_train_preds_rf,
            output_dict=True))
    plot_df_helper(rf_train_report, 'rf_train_report', constants.RESULTS_OUT_PATH)

    lr_test_report = pd.DataFrame.from_dict(
        classification_report(
            y_test,
            y_test_preds_lr,
            output_dict=True))
    plot_df_helper(lr_test_report, 'lr_test_report', constants.RESULTS_OUT_PATH)

    lr_train_report = pd.DataFrame.from_dict(
        classification_report(
            y_train,
            y_train_preds_lr,
            output_dict=True))
    plot_df_helper(lr_train_report, 'lr_train_report', constants.RESULTS_OUT_PATH)


def plot_roc_curve(models, X_test, y_test, file_path):
    '''
    Helper function to plot roc curve for each model in one plot

    Input:
            models: models to plot roc curve
            model_names: model names for each model
            X_test: X testing data
            y_test: y testing data
            file_path: path to save figure

    Output:
            None
    '''
    plt.figure(figsize=(15, 8))
    ax = plt.gca()

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    for model, model_name in zip(models, constants.MODEL_NAMES):
        y_pred = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        auc = round(roc_auc_score(y_test, y_pred), 4)
        plt.plot(fpr, tpr, label=model_name + ", AUC=" + str(auc))
        ax.legend()

    file_path += 'roc_auc.png'
    plt.savefig(file_path)


def feature_importance_plot(model, X_data, file_path):
    '''
    Creates and stores the feature importances in pth
    Input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            file_path: path to store the figure

    Output:
            None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    file_path += 'feature_importance.png'
    plt.savefig(file_path)


def train_models(X_train, X_test, y_train, y_test, file_path):
    '''
    Train, store model results: images + scores, and store models
    Input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
              file_path: path to save file
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    
    joblib.dump(cv_rfc.best_estimator_, constants.MODELS_OUT_PATH+'rfc_model.pkl')
    joblib.dump(lrc, constants.MODELS_OUT_PATH+'logistic_model.pkl')

    rfc_model = joblib.load(constants.MODELS_OUT_PATH+'rfc_model.pkl')
    lr_model = joblib.load(constants.MODELS_OUT_PATH+'logistic_model.pkl')

    models = [lr_model, rfc_model]
    plot_roc_curve(models, X_test, y_test, file_path)

    # CHECK IF MODEL HAS FEATURE_IMPORTANCE 
    feature_importance_plot(rfc_model, X_test, file_path)


def main():
    DF = import_data(constants.DF_PATH)

    # DF[constants.RESPONSE] = DF['Attrition_Flag'].apply(
    #     lambda val: 0 if val == "Existing Customer" else 1)
    # DF = request.config.cache.get('cache_df', None)

    perform_eda(DF)

    X_train, X_test, y_train, y_test = perform_feature_engineering(
        DF, constants.CATEGORY_LST, constants.RESPONSE)

    train_models(X_train, X_test, y_train, y_test, constants.RESULTS_OUT_PATH)


if __name__ == "__main__":
    main()
