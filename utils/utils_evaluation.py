#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions for model evaluation"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score, classification_report
from sklearn.inspection import permutation_importance
from scipy import stats

import utils.settings as s

s.init()
seed = s.seed

def mean_bias_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = y_true.reshape(len(y_true),1)
    y_pred = y_pred.reshape(len(y_pred),1)   
    diff = (y_true-y_pred)
    mbe = diff.mean()

    return mbe

def normalized_root_mean_squared_error(y_true, y_pred):
    squared_error = np.square((y_true - y_pred))
    sum_squared_error = np.sum(squared_error)
    rmse = np.sqrt(sum_squared_error / y_true.size)
    nrmse_loss = rmse/np.std(y_pred)
    return nrmse_loss

def mean_absolute_percentage_error(y_true, y_pred): 
    """" Calculate  MAPE from predicted and actual target  """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return 1/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)) * 100)

def root_mean_squared_error(y_true, y_pred):
    return  np.sqrt( np.mean((y_true - y_pred)**2) )
   
def evaluation_report(y_true, y_pred): 
    """
    Print model performance evaluation between predicted and actual target
    y_true : actual y 
    y_pred : predicted y
    #return : evaluation metrics:  mse, rmse, mbe, mape, r2
    """
    rmse = root_mean_squared_error(y_true, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)#np.mean((np.abs(y_true - y_pred)**2))
    mbe = mean_bias_error(y_true, y_pred)
    r2c = r2_score(y_true, y_pred)

    print(
    f"""Model Performance:
        Root Mean Square Error: {round(rmse,2)}
        Symmetric Mean Abs. Percentage Error: {round(smape,2)}
        Mean Absolute Error: {round(mae,2)}
        Mean Bias Error: {round(mbe,2)}
        R²-Score: {round(r2c,3)}
    """
    )
#    return mse, rmse, mae, mape, r2c


def compute_score(y_true, y_pred):
    """
    https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html
    """
    return {
        "R2": f"{r2_score(y_true, y_pred):.3f}",
        "MedAE": f"{median_absolute_error(y_true, y_pred):.3f}", # TODO check pros compared to MAE
    }


def empirical_vs_predicted(X_test, y_test, models_list):
    """
    models_list (list): in the order [model_notransform, model_log, model_quantile, model_boxcox, model_sqrt]  # TODO robustify, remove hardcodes
    return df with statistics
    """
    empirical_vs_predicted = [] 

    for idx, test_set in enumerate([y_test, models_list[0].predict(X_test), models_list[1].predict(X_test), models_list[2].predict(X_test), models_list[3].predict(X_test), models_list[4].predict(X_test)]):
        test_statistics = stats.describe(test_set) 
        empirical_vs_predicted.append(
            pd.Series(
            {
                'nobs':  test_statistics[0],
                'median': np.median(test_set), #round(test_statistics[2], 2),
                'mean':  np.mean(test_set),# round(test_statistics[3], 2),
                'min max':  [test_statistics[1][0], test_statistics[1][1]],
                'variance': round(test_statistics[4], 2),
            }
            )
        )
    df_stats = pd.DataFrame(
        empirical_vs_predicted,
        index=(
            ["empirical", "no transform", "natural log", "quantile", "box-cox", "sqrt"]
        )
    )
    return df_stats


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

