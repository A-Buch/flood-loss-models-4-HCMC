#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions for model evaluation"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.model_selection import RepeatedKFold, cross_validate, cross_val_predict

from scipy import stats

# load r library initally
#%load_ext rpy2.ipython

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri # pandas.DataFrames to R dataframes 

pandas2ri.activate()
rpy2.ipython.html.init_printing()

# get R packages
base = importr("base")
nestedcv = importr("nestedcv")
permimp = importr("permimp")  # conditional permutation feature importance
pdp = importr("pdp")

# pandas.DataFrames to R dataframes 
from rpy2.robjects import pandas2ri, Formula
pandas2ri.activate()

# print r df in html
import rpy2.ipython.html
rpy2.ipython.html.init_printing()


# get libraries for CRF processing, ctree_controls etc
permimp = importr('permimp')  # conditional permutation feature importance
caret = importr('caret') # package version needs to be higher than  >=  6.0-90


import utils.settings as s
import utils.utils_feature_selection as fs

s.init()
seed = s.seed

def mean_bias_error(y_true, y_pred):
    """" Calculate MBE from predicted and actual target  """
    #y_true = np.array(y_true)
    #y_pred = np.array(y_pred)
    #y_true = y_true.reshape(len(y_true),1)
    #y_pred = y_pred.reshape(len(y_pred),1)   
    return (y_true-y_pred).mean()

def mean_absolute_percentage_error(y_true, y_pred): 
    """" Calculate MAPE from predicted and actual target  """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """" Calculate SMAPE from predicted and actual target  """
    return 1/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)) * 100) 

def root_mean_squared_error(y_true, y_pred):
    """" Calculate RMSE from predicted and actual target  """
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
        "MAE": f"{mean_absolute_error(y_true, y_pred):.3f}",
        "RMSE": f"{root_mean_squared_error(y_true, y_pred):.3f}",
    }


# class Model(object):
#     """  
#     Parent class for  model fitting and model evaluation
#     """
#     def __init__(self, inner_cv, outer_cv):
#            self.inner_cv = inner_cv


class ModelEvaluation(object):
    """
    
    """        
    def __init__(self, models_trained_ncv, Xy, target_name, score_metrcis, seed):
        #super(model_fitting, self).__init__()
        self.models_trained_ncv = models_trained_ncv
        self.X = pd.DataFrame(Xy.drop(target_name, axis=1))
        self.X = pd.DataFrame(
                MinMaxScaler().fit_transform(self.X),   
                columns=self.X.columns) 
        self.y: pd.DataFrame = Xy[target_name]
        self.outer_cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=seed)
        self.k_folds:int = 10
        self.score_metrcis = score_metrcis
        self.seed: int = seed


    def model_evaluate_ncv(self):
        """  
        
        """    
        ## predict y on outer folds of nested cv
        y_pred = cross_val_predict(
            self.models_trained_ncv,  # estimators from inner cv
            self.X, self.y,
            cv=self.k_folds, # KFold without repeats to have for each sample one predicted value 
            method="predict"
        )

        ## get generalization performance on outer folds of nested cv
        model_performance_ncv = cross_validate(
            self.models_trained_ncv, 
            self.X, self.y, 
            scoring=self.score_metrcis,  # Strategies to evaluate the performance of the cross-validated model on the test set.
            cv=self.outer_cv, 
            return_estimator=True,
        ) 
        print(
            "model performance for MAE (std) of outer CV: %.3f (%.3f)"%(
                model_performance_ncv["test_MAE"].mean(), np.std(model_performance_ncv["test_MAE"])
            ))
        
        return y_pred, model_performance_ncv
    
    def r_model_evaluate_ncv(self, models_trained_ncv):
        """ 
        """            
        ## predict y based on outer folds of nested cv
        y_pred = nestedcv.train_preds(models_trained_ncv) 

        ## get generalization performance on outer folds of nested cv
        model_performance_ncv = {
            "test_MAE_old" : base.summary(models_trained_ncv)[3][2],
            "test_MAE" : mean_absolute_error(self.y, y_pred),
            "test_RMSE" : root_mean_squared_error(self.y, y_pred),
            "test_MBE" : mean_bias_error(self.y, y_pred),
            "test_R2": r2_score(self.y, y_pred),
            "test_SMAPE" : symmetric_mean_absolute_percentage_error(self.y, y_pred),
        }
        return y_pred, model_performance_ncv


    def permutation_feature_importance(self, final_model, Xy, target_name, repeats=10, seed=seed):
    #def permutation_feature_importance(model, X_test, y_test, y_pred, criterion= r2_score):
        """
        Calculate permutation based feature importance , the importance scores represents the increase in model error
        final_model : final sklearn model
        Xy : pd.Dataframe with predictors and response variable <target_name>
        target_name (str): name of response variable 
        criterion : sklearn evaluation metrics, default r2_score 
        
        return: pd DataFrame with importance scores
        """
        permutation_fi = permutation_importance(final_model, Xy.drop(target_name, axis=1), Xy[target_name], n_repeats=repeats, random_state=seed)

        return permutation_fi.importances_mean, permutation_fi.importances_std, permutation_fi.importances


    def r_permutation_feature_importance(self, final_model):  # final_model):
        # sourcery skip: inline-immediately-returned-variable
        """  
        """ 
        importances = permimp.permimp(
            self, final_model, threshold=0.95, conditional=True, progressbar=False
        )
        return importances

    
    def empirical_vs_predicted(self, X_test, y_test, models_list):
        """
        models_list (list): in the order [model_notransform, model_log, model_quantile, model_boxcox, model_sqrt]  # TODO robustify, remove hardcodes
        return df with statistics
        """
        empirical_vs_predicted = [] 

        for test_set in [y_test, models_list[0].predict(X_test), models_list[1].predict(X_test), models_list[2].predict(X_test), models_list[3].predict(X_test), models_list[4].predict(X_test)]:
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
        return pd.DataFrame(
            empirical_vs_predicted,
            index=(
                [
                    "empirical",
                    "no transform",
                    "natural log",
                    "quantile",
                    "box-cox",
                    "sqrt",
                ]
            ),
        )

    ## @decorator(model=final_models_trained["crf"], Xy=eval_set_list["crf"]["crf"], target_name=target, feature_name="flowvelocity", scale=True) 
    ## not using decorator @
    def get_partial_dependence(self, **kwargs):
        model= kwargs["model"]
        Xy = kwargs["Xy"]
        y_name = kwargs["y_name"]
        feature_name = kwargs["feature_name"]
        scale = kwargs["scale"]

        X = Xy.dropna().drop(y_name, axis=1)

        # scale feature distributions in pd plots across models
        if scale:
            X =  pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)

        partial_dep = partial_dependence(   
            model, X=X, features=feature_name, 
            grid_resolution=X.shape[0], kind="average", #**further_params,
        )
        return pd.DataFrame(
            {
                feature_name: partial_dep.grid_values[0],
                "yhat": partial_dep.average[0],
            }
        )


    ## decorator for R model
    def decorator_func(self, model , Xy, y_name, feature_name, scale=True):
        """
        Decorator to get partial dependence instead of python-sklearn-model from R-party-model
        """
        def r_get_partial_dependence(func):
            def wrapper(*args, **kwargs):
    
                X = Xy.dropna().drop(y_name, axis=1)
                Xy = pd.concat([Xy[y_name], X], axis=1)

                robjects.r('''
                    r_partial_dependence <- function(model, df, predictor_name, verbose=FALSE) {
                        pdp::partial(model, train=df, pred.var=predictor_name, type="regression", plot=FALSE )  
                    }
                ''') #  , plot=FALSE --> to get pdp values
                r_partial_dependence = robjects.globalenv['r_partial_dependence'] 
                
                partial_dep = r_partial_dependence(model, Xy, feature_name)
                
                return fs.r_dataframe_to_pandas(partial_dep)
            
            return wrapper
        return r_get_partial_dependence
    


# def r_models_cv_results(model):
#     """
#     Get training results for all tested model settings during CV and tunning in R
#     """
#     robjects.r('''
#         r_models_cv_results <- function(m, verbose=FALSE) {
#             m$results
#         }
#     ''')
#     r_models_cv_results = robjects.globalenv['r_models_cv_results']
#     return (r_models_cv_results(model))
    

def r_models_cv_predictions(model, idx=0):
    """
    Get y_pred and y_true for a certain model during CV in R
    model : R model from nestedcv.train()
    idx (int): index position of trained model from inner cv
    return: pandas Dataframe with y_pred and y_test values 
    """
    robjects.r('''
        r_models_cv_predictions <- function(m, idx, verbose=FALSE) {
            m$outer_result[[idx]]$preds
        }
    ''') 
    r_model_prediction = robjects.globalenv['r_models_cv_predictions']

    return fs.r_dataframe_to_pandas(r_model_prediction(model, idx))



