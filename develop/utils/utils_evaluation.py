#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions for model evaluation"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
from sklearn.inspection import permutation_importance

import utils.settings as s

s.init()
seed = s.seed



def mean_absolute_percentage_error(y_true, y_pred): 
    """" Calculate  MAPE from predicted and actual target  """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluation_report(y_true, y_pred): 
    """
    Print model performance evaluation between predicted and actual target
    y_true : actual y 
    y_pred : predicted y
    #return : evaluation metrics:  mse, rmse, mae, mape, r2
    """
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)#np.mean((np.abs(y_true - y_pred)**2))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2c = r2_score(y_true, y_pred)

    print(f"""
    Model Performance:
        Mean Squared Error: {round(mse,1)}
        Root Mean Square Error: {round(rmse,1)}
        Mean Absolute Error: {round(mae,1)}
        Mean Absolute Percentage Error: {round(mape,1)}
        R²-Score: {round(r2c,1)}
    """
    )
#    return mse, rmse, mae, mape, r2c



def permutation_feature_importance(model, X_test, y_test, repeats=10, seed=seed):
#def permutation_feature_importance(model, X_test, y_test, y_pred, criterion= r2_score):
    """
    Calculate permutation based feature importance , the importance scores represents the increase in model error
    model : trained sklearn model (but not applied on test set)
    X_test : pdDataframe with independend features from test set
    y_test : pd.Series with target values from test set
    y_pred : pd.Series with predicted target values (predicted based on X_test)
    criterion : sklearn evaluation metrics, default r2_score 
    
    return: pd DataFrame with importance scores
    """
    permutation_fi = permutation_importance(model, X_test, y_test, n_repeats=repeats, random_state=seed)

    ## self made without multiple repeats  ##
    # permutation_fi_matrix = []
    # original_error = criterion(y_test, y_pred) 
    # for feature in X_test.columns:
    #     perbutated_data= X_test  # copy.deepcopy(X_test)
    #     perbutated_data[f"{feature}"] = np.random.permutation(perbutated_data.loc[ : , feature])
    #     perbutated_pred = model.predict(perbutated_data)
    #     perbutated_error = criterion(y_test, perbutated_pred)#criterion(y_test, perbutated_pred)
    #     permutation_fi_matrix.append((original_error - perbutated_error))    
    # permutation_fi = pd.DataFrame(permutation_fi_matrix, index=X_test.columns, columns=['importances'])#.transpose()

    return permutation_fi.importances_mean, permutation_fi.importances_std, permutation_fi.importances
