#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions"""

__author__ = "Anna Buch, Heidelberg University"
__email__ = "a.buch@stud.uni-heidelberg.de"


import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri # pandas.DataFrames to R dataframes 
from rpy2.robjects.conversion import localconverter

import utils.settings as s

logger = s.init_logger("__feature_selection__")

pandas2ri.activate()
# rpy2.ipython.html.init_printing()

stats = importr("stats")




def equal_freq_binning(df, variable_name, cuts=3, group_labels=None, drop_old_variable=False):
    """
    Split variable into cateogries, each category with equal number of data points
    df : pandas dataframe
    variable_name (str): variable name, this variable is discretized
    cuts (int): number of categories
    group_labels (list): list of length of category number
    return: Dataframe with new discretized variable based on euqal number of datapoints per category
    """
    if group_labels is None:
        group_labels = ["low", "medium", "high"]
    logger.info(f"{df.shape[0]} records are euqally split into categories, so that same number of records is in each class (equal frequency binning) ")
    logger.info(f"Group labels and bins : {group_labels, pd.qcut(df[variable_name], q=cuts).value_counts()}")

    new_variable_name = variable_name + "_c"
    try:
        df[new_variable_name] = pd.qcut(df[variable_name], q=cuts, labels=group_labels)
    except Exception:
        logger.info("drop dublicates")
        df[new_variable_name] = pd.qcut(df[variable_name], q=cuts, duplicates="drop")

    if drop_old_variable is True:
        df = df.drop(variable_name, axis=1)

    return df


def normalize_X(X_train, X_test):
    """
    Normalize X_train and then fit scaler to X_test based on MinMaxScaler
    X_train (df): pd.DataFrame with predictors for training
    X_test (df): pd.DataFrame with predictors for testing
    return: scaled X_train and X_test
    """
    ## normalize X data 
    scaler_for_X = MinMaxScaler().fit(X_train)
    X_train_scaled = scaler_for_X.transform(pd.DataFrame(X_train))
    X_test_scaled = scaler_for_X.transform(pd.DataFrame(X_test))

    return pd.DataFrame(X_train_scaled, columns=X_train.columns), pd.DataFrame(X_test_scaled, columns=X_test.columns)
    ## normalize y data 
    ## TODO find and test a better way to scale y, eg via TransformedTargetRegressor() ?
    # y_train =np.array(y_train).reshape(-1, 1)
    # y_test = np.array(y_test).reshape(-1, 1)
    # scaler_for_y = MinMaxScaler().fit(y_train)
    # y_train = scaler_for_y.transform(pd.DataFrame(y_train))
    # y_test = scaler_for_y.transform(pd.DataFrame(y_test))


def r_ctree_statistics(model):
    """
    model = uses r model inside python of type rpy2.robjects.vectors.ListVector
    return: pandas dataframe with log p-values
    ## Code snippets: 
    ## https://stats.stackexchange.com/questions/171301/interpreting-ctree-partykit-output-in-r, 
    ## https://www.askpython.com/python/examples/r-in-python
    """
    robjects.r('''
        func_stats <- function(m, verbose=FALSE) {
            stats = nodeapply(m, ids=1, function(n) info_node(n)$criterion)
            stats = stats$`1`            
            }
        ''')
    
    # get function outside R
    func_stats = robjects.globalenv['func_stats'] 

    #  store statistics in pd df
    df_ctree_stats = pd.DataFrame()

    df_ctree_stats["statistic"] = func_stats(model)[0]
    df_ctree_stats["p_value"] = func_stats(model)[1]
    df_ctree_stats["criterion"] = func_stats(model)[2]
    df_ctree_stats = df_ctree_stats.T

    return df_ctree_stats


def r_best_hyperparamters(model):
    """
    Get hyperparamters from best model in R GridSearch 
    """
    robjects.r('''
        r_best_hyperparamters <- function(m, verbose=FALSE) {
            m$bestTune
        }
    ''')
    r_best_hyperparamters = robjects.globalenv['r_best_hyperparamters']
    return r_best_hyperparamters(model)


def r_dataframe_to_pandas(df):
    """
    Convert a R DataFrame to a Pandas DataFrame by keeping column names
    df : R DataFrame 
    return: pandas DataFrame
    """
    with localconverter(robjects.default_converter + pandas2ri.converter):
        pd_dataframe = robjects.conversion.rpy2py(df)
    return pd_dataframe


def vif_score(X_scaled_drop_nan):
    df_vif = pd.DataFrame()
    df_vif["names"]  = X_scaled_drop_nan.columns
    df_vif["vif_scores"] = [
        variance_inflation_factor(X_scaled_drop_nan.values.astype(float), i)
        for i in range(len(X_scaled_drop_nan.columns))
    ]
    df_vif = df_vif.sort_values("vif_scores", ascending=False).reset_index(drop=True)
    logger.info(f"averaged VIF score is around: {round(df_vif.vif_scores.mean(),1)}")

    return df_vif


def normalize_feature_importances(df_feature_importances, scale_range=(0,10)):
    """ 
    Normalize columns of pd.DatFrame to same scale
    scale_range (tuple of integers): range of scale (min, max) 
    return: scaled pd.DataFrame 
    """
    logger.info(f"Normalize columns to scale: {scale_range[0]} - {scale_range[1]}")
    ## scale importance scores to  same units (non important feautres were removed before)
    df_feature_importances = pd.DataFrame(
        MinMaxScaler(feature_range=scale_range).fit_transform(df_feature_importances), 
        index=df_feature_importances.index,
        columns=df_feature_importances.columns
    )
    return df_feature_importances


def calc_weighted_sum_feature_importances(df_feature_importances, model_weights):
    """ 
    model_weights (dict) : keys are feature importnace columns, values are the weights 
    return: pd.DataFrame same as df_feature_importances 
    but added column with weighted sum for each feature importance
    """
    ## Normalize feature importnaces to same scale
    df_feature_importances = normalize_feature_importances(df_feature_importances)

    ## assigne weights to importnace scores; weight better models stronger
    models_fi_list = []
    for model_fi, weight in model_weights.items(): 
        model_fi.split("_")[0]
        model_fi_weighted = f"{model_fi}_weighted"
        df_feature_importances[model_fi_weighted] =  df_feature_importances[model_fi] / weight
        models_fi_list.append(model_fi_weighted)

    ## derive weighted sum for each feature across all models
    ## TODO remove hardcode, make flexible to different number of models
    df_feature_importances["weighted_sum_importances"] = df_feature_importances[models_fi_list].fillna(0).sum(axis=1)

    return df_feature_importances.sort_values("weighted_sum_importances", ascending=True)


def save_selected_features(X_train, y_train, selected_feat_cols, filename="fs_model.xlsx"):
    """
    Selects feautres from training set and saves them in excel file
    X_train (df): X training set with predictors
    y_train (df): y training set with target
    selected_feat_cols (list): column names of selected features
    """
    selected_feat = X_train.loc[:,selected_feat_cols]
    not_selected_feat = X_train.drop( selected_feat, axis=1)

    logger.info(f"total features: {X_train.shape[1]}")
    logger.info(f"dropped features: {len(not_selected_feat.columns)}")
    logger.info(f"selected {len(selected_feat_cols)} features: \n{X_train[selected_feat_cols].columns.to_list()}\n")  # noqa: E501

    ## write selected features from training set to disk
    train = pd.concat([y_train, X_train], axis=1)
    df = train[ y_train.columns.to_list() + selected_feat_cols.to_list() ]

    logger.info(f"Saving selected features to disk: {filename}")
    df.to_excel(filename, index=False)


