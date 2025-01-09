#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Metrics for model evaluation"""

__author__ = "Anna Buch, Heidelberg University"
__email__ = "anna.buch@uni-heidelberg.de"


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats

import settings as s

logger = s.init_logger("__evaluation_utils__")


def mean_bias_error(y_true, y_pred):
    """ " Calculate MBE from predicted and actual target"""
    return (y_pred - y_true).mean()


# def mean_absolute_percentage_error(y_true, y_pred):
#     """" Calculate MAPE from predicted and actual target  """
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """ " Calculate SMAPE from predicted and actual target"""
    return 100 / len(y_true) * np.sum(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))
    # return 100/len(y_true) * np.sum(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)) )


def root_mean_squared_error(y_true, y_pred):
    """ " Calculate RMSE from predicted and actual target"""
    # (np.mean((y_pred-y)**2))**(1/2)
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def coef_variation(x):
    """coefficient of variation"""
    return np.std(x, ddof=1) / np.mean(x) * 100


def empirical_vs_predicted(y_true, y_pred):
    """
    return (pd.DataFrame): with statistics of predicted and observed target values
    """
    empirical_vs_predicted = []

    for y_set in [y_true.astype(int), y_pred.astype(int)]:
        test_statistics = stats.describe(np.array(y_set))

        empirical_vs_predicted.append(
            pd.Series(
                {
                    "nobs": test_statistics[0],
                    "median": np.median(y_set),
                    "mean": np.mean(y_set),
                    "min max": [test_statistics[1][0], test_statistics[1][1]],
                    "variance": round(test_statistics[3], 2),
                    "standard deviation": round(np.std(y_set), 2),
                    "coef. of variation": pd.DataFrame(y_set).apply(coef_variation)[0],
                }
            )
        )
    return pd.DataFrame(empirical_vs_predicted, index=(["empirical", "predicted"]))


def calc_confidence_interval(y_set, confidence_level=0.95):
    """ """
    conf_interval = stats.t.interval(confidence=confidence_level, df=len(y_set) - 1, loc=np.mean(y_set), scale=stats.sem(y_set))
    ## alternative for larger ds and normal distirbution
    # stats.norm.interval(confidence=0.95,
    #                  loc=np.mean(y_set),
    #                  scale=stats.sem(y_set))

    counts = ((y_set > conf_interval[0]) & (y_set <= conf_interval[1])).sum(axis=0)
    logger.info(f"confidence interval: {conf_interval},\ncases within interval: {counts}")

    return np.round(conf_interval, 2)


def reverse_probability_scores(df, predproba_colname="y_proba", pred_colname_value=("y_pred", 0.0)):
    ## TODO make description shorter
    ## TODO catch y_pred, y_proba from model_evaluate_ncv
    """
    Set probability values returned from sklearn.cross_val_predict() in respect to predicted target (here binary class of damage and no-damage)
    Currently zero-loss cases have also probability between 0.5 - 1.0 % --> should be 0.0 - 0.49 % for zero-loss; damage-cases with 0.50-1.0%
    Note: Proabiliites have to be in range [0,1], only for binary classification tasks
    Example: if ypred == 1, do nothing; if ypred == 0 --> then proba should be 1.0 -> 0.0 ; 0.98 -> 0.02  [old proba -> new proba]

    df : pd.DataFrame with probability column and binary prediction column
    predproba_colname (str): column name indicating column with probabilities that should be converted
    pred_colname_value (tuple: str, float): column name of binary predictions and value in this column indicating which probabilities in predproba_colname should be converted
    return df with reversed probabilities in respective column
    """
    df[predproba_colname] = df.apply(
        lambda x: np.abs(x[predproba_colname] - 1)
        if x[pred_colname_value[0]] == pred_colname_value[1]
        else x[predproba_colname],  # else x["y_proba"] -> dont do anything for predicted damage-cases
        axis=1,
    ).round(3)

    return df


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
