#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions"""


import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# load r library initally
#%load_ext rpy2.ipython
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri # pandas.DataFrames to R dataframes 
from rpy2.robjects.conversion import localconverter
import rpy2.ipython.html # print r df in html

pandas2ri.activate()
rpy2.ipython.html.init_printing()

# get basic R packages
utils = importr('utils')
base = importr('base')
dplyr = importr('dplyr')
stats = importr("stats")
# get partykit library containing ctree , ctree_controls etc
partykit = importr('partykit')
party = importr('party')



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


def r_dataframe_to_pandas(df):
    """
    Convert a R DataFrame to a Pandas DataFrame by keeping column names
    df : R DataFrame 
    return: pandas DataFrame
    """
    with localconverter(robjects.default_converter + pandas2ri.converter):
        pd_dataframe = robjects.conversion.rpy2py(df)

    return pd_dataframe


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
    return (r_best_hyperparamters(model))


def r_models_cv_results(model):
    """
    Get training results for all tested model settings during CV and tunning in R
    """
    robjects.r('''
        r_models_cv_results <- function(m, verbose=FALSE) {
            m$results
        }
    ''')
    r_models_cv_results = robjects.globalenv['r_models_cv_results']
    return (r_models_cv_results(model))
    



def kfold_cross_validation():
    """    
    """


def save_selected_features(X_train, y_train, selected_feat_cols, filename=f"fs_model.xlsx"):
    """
    Selects feautres from training set and saves them in excel file
    X_train (df): X training set with predictors
    y_train (df): y training set with target
    selected_feat_cols (list): column names of selected features
    """
    selected_feat = X_train.loc[:,selected_feat_cols]
    not_selected_feat = X_train.drop( selected_feat, axis=1)

    print("total features: {}".format((X_train.shape[1])))
    print("selected features: {}".format(len(selected_feat_cols)))
    print("dropped features: {}".format(len(not_selected_feat.columns)))
    #print("selected features: \n{}\n".format(X_train[selected_feat_cols].columns.to_list()))
    #print("dropped features: \n{}\n".format(X_train[not_selected_feat.columns].columns.to_list()))

    ## write selected features from training set to disk
    train = pd.concat([y_train, X_train], axis=1)
    df = train[ y_train.columns.to_list() + selected_feat_cols.to_list() ]
    #df_elastic_net.info()

    print(f"Saving model to disk: {filename}")
    df.to_excel(filename, index=False)



# def r_ctree_p(model):
#     """
#     model = uses r model inside python of type rpy2.robjects.vectors.ListVector
#     return: pandas dataframe with p-values    
#     ## Code snippets: 
#     ## https://stats.stackexchange.com/questions/171301/interpreting-ctree-partykit-output-in-r, 
#     ## https://www.askpython.com/python/examples/r-in-python#     """
#     robjects.r('''
#         func_p <- function(m, verbose=FALSE) {
#             nodeapply(m, ids=nodeids(m), function(n) info_node(n)$p.value)
#         }
#         ''')
#     func_p = robjects.globalenv['func_p'] # get r function outside r env



