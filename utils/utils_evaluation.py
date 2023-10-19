#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions for model evaluation"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance, partial_dependence
from scipy import stats

# load r library initally
#%load_ext rpy2.ipython

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri # pandas.DataFrames to R dataframes 

pandas2ri.activate()
rpy2.ipython.html.init_printing()
pdp = importr("pdp")

import utils.settings as s
import utils.utils_feature_selection as fs


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
        "MAE": f"{mean_absolute_error(y_true, y_pred):.3f}",
        "RMSE": f"{root_mean_squared_error(y_true, y_pred):.3f}",
    }


class model_evaluation(object):
    """
    
    """
    
    def empirical_vs_predicted(self, X_test, y_test, models_list):
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

    def permutation_feature_importance(self, model, X_test, y_test, repeats=10, seed=seed):
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


    ## decorator for R model
    def decorator_func(self, model , Xy, y_name, feature_name, scale=True):
        """
        Decorator to get partial dependence instead of python-sklearn-model from R-party-model
        """
        def r_get_partial_dependence(func):
            def wrapper(*args, **kwargs):
    
                X = Xy.dropna().drop(y_name, axis=1)
        
                # scaled feature distributions in pd plots across models
                if scale:
                    X = pd.DataFrame(
                        MinMaxScaler().fit_transform(X),
                        columns=X.columns
                    )
                partial_dep = r_partial_dependence(
                    model, 
                    Xy,
                    feature_name
                    )
                #return func(*args, **kwargs)
                return fs.r_dataframe_to_pandas(partial_dep)
            
            return wrapper
        return r_get_partial_dependence


    #@decorator(model=final_models_trained["crf"], Xy=eval_set_list["crf"]["crf"], target_name=target, feature_name="flowvelocity", scale=True) 
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
            X =  pd.DataFrame(
                MinMaxScaler().fit_transform(X),
                columns=X.columns
            )
        partial_dep = partial_dependence(   
            model,
            X=X,
            features=feature_name,
            grid_resolution=X.shape[0],
            kind="average", 
            #**further_params,
        )
        partial_dep_df = pd.DataFrame({
            feature_name : partial_dep.grid_values[0],
                "yhat": partial_dep.average[0]
            }
        )
        return partial_dep_df


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


def r_partial_dependence(model, df, predictor_name:str):
    """
    Get partial dependence for single predictor from fitted R model
    model : fitted R model eg. from nestedcv.train()
    df: pandas DataFrame with target and features used to fit model
    predictor_name (str): Name of predictor
    return: pandas Dataframe with gridvalues [yhat] and partial dependences
    """
    robjects.r('''
        r_pdp <- function(m, df, predictor_name, verbose=FALSE) {
            pdp::partial(
                m, 
                train=df,
                pred.var=predictor_name,
                type="regression",
                plot=FALSE
            )  
        }
        ''') #  , plot=FALSE --> to get pdp values
    r_pdp = robjects.globalenv['r_pdp'] 
    return r_pdp(model, df, predictor_name)
    

