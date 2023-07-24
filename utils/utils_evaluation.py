#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions for model evaluation"""

import numpy as np
import math 
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score, classification_report




def mean_absolute_percentage_error(y_true, y_pred): 
    """" Calculate  MAPE from predicted and actual target  """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluation_report(y_true, y_pred): 
    """
    Print model performance evaluation between predicted and actual target
    """
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean((np.abs(y_true - y_pred)**2))
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
    return mse, rmse, mae, mape, r2c