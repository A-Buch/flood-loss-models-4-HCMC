#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Metrics for model evaluation"""

__author__ = "Anna Buch, Heidelberg University"
__email__ = "a.buch@stud.uni-heidelberg.de"


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats

import utils.settings as s

logger = s.init_logger("__evaluation_metrics__")



def mean_bias_error(y_true, y_pred):
    """" Calculate MBE from predicted and actual target  """
    return (y_true-y_pred).mean()


def mean_absolute_percentage_error(y_true, y_pred): 
    """" Calculate MAPE from predicted and actual target  """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """" Calculate SMAPE from predicted and actual target  """
    # return 1/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)) * 100) 
    # return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)) ) 
    return 100/len(y_true) * np.sum(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)) ) 


def root_mean_squared_error(y_true, y_pred):
    """" Calculate RMSE from predicted and actual target  """
    return  np.sqrt( np.mean((y_true - y_pred)**2) )
   
 
def empirical_vs_predicted(y_true, y_pred):
    """
    return (pd.DataFrame): with statistics of predicted and observed target values
    """
    empirical_vs_predicted = [] 

    for y_set in [y_true.astype(int), y_pred.astype(int)]:
        test_statistics = stats.describe(np.array(y_set))
        empirical_vs_predicted.append(
            pd.Series({
                'nobs':  test_statistics[0],
                'median': np.median(y_set),
                'mean':  np.mean(y_set),
                'min max':  [test_statistics[1][0], test_statistics[1][1]],
                'variance': round(test_statistics[3], 2),
            })
        )
    return pd.DataFrame(empirical_vs_predicted, index=(["empirical", "predicted"]))

  
        
def evaluation_report(y_true, y_pred): 
    """
    Print model performance evaluation between predicted and actual target
    y_true : actual y 
    y_pred : predicted y
    return : evaluation metrics:  mse, rmse, mbe, mape, r2
    """
    rmse = root_mean_squared_error(y_true, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mbe = mean_bias_error(y_true, y_pred)
    r2c = r2_score(y_true, y_pred)

    logger.info(
    f"""Model Performance:
        Root Mean Square Error: {round(rmse,2)}
        Symmetric Mean Abs. Percentage Error: {round(smape,2)}
        Mean Absolute Error: {round(mae,2)}
        Mean Bias Error: {round(mbe,2)}
        R²-Score: {round(r2c,3)}
    """
    )


def compute_score(y_true, y_pred):
    """
    https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html
    """
    return {
        "MAE": f"{mean_absolute_error(y_true, y_pred):.3f}",
        "RMSE": f"{root_mean_squared_error(y_true, y_pred):.3f}",
    }
