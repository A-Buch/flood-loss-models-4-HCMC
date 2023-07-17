#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions"""


import pandas as pd

# load r library initally
#%load_ext rpy2.ipython

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri # pandas.DataFrames to R dataframes 
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
    print("selected features: \n{}\n".format(X_train[selected_feat_cols].columns.to_list()))
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



