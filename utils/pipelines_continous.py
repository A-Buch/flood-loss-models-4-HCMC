#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pipelines for feature selection"""

import joblib
import numpy as np

from sklearn.linear_model import ElasticNet
from sklearn.ensemble import BaggingRegressor
from xgboost import XGBRegressor

from sklearn.pipeline import Pipeline
# from imblearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, quantile_transform, PowerTransformer, power_transform
from sklearn.compose import TransformedTargetRegressor

import utils.settings as s


s.init()
seed = s.seed

def main():

    ############  Elastic Net  ##################
    pipe_en = Pipeline( steps = [ ('model', ElasticNet()) ] )

    # Elastic Net with Bagging
    ensemble_model = {
        'model': BaggingRegressor,   # default bootstrap=True
        'kwargs': {'estimator': ElasticNet(),  # estimator -> variable from Bagging Regressor
                   'bootstrap': True,
                   #'random_state':seed
                  }  # TODO: pass 'random_state':seed to baggingregressor
    }
    pipe_bag_en = Pipeline([
        ('bagging', ensemble_model['model'] (**ensemble_model['kwargs']) )
    ])

    ### Transformation: natural log
    pipe_en_log = Pipeline([
        ('model', TransformedTargetRegressor(regressor=ElasticNet(),
            func=np.log1p, inverse_func=np.expm1)
        )])  # np.expm1 = provides greater precision than exp(x) - 1 for small values of x.

    ###  Transformation: quantile
    pipe_en_quantile = Pipeline([
                ('model', TransformedTargetRegressor(regressor=ElasticNet(),
                    transformer=QuantileTransformer(n_quantiles=900, output_distribution="normal"))
                )])

    ###  Transformation: Box-cox 
    pipe_en_boxcox = Pipeline([
                ('model', TransformedTargetRegressor(regressor=ElasticNet(),
                    transformer=PowerTransformer(method="box-cox", standardize=False) # def=False:
                    #func=np.reciprocal, inverse_func=np.expm1
                    )
                )])

    ###  Transformation: square root 
    pipe_en_sqrt = Pipeline([
        ('model', TransformedTargetRegressor(regressor=ElasticNet(),
            func=np.sqrt, inverse_func=np.square #np.expm1
        )
        )])



    ############  XGBoost Regressor  ##################
    pipe_xgb = Pipeline( steps = [ ('model', XGBRegressor()) ] )

    ### Transformation: natural log
    pipe_xgb_log = Pipeline([
        ('model', TransformedTargetRegressor(regressor=XGBRegressor(),
            func=np.log1p, inverse_func=np.expm1)
        )])  # np.expm1 = provides greater precision than exp(x) - 1 for small values of x.

    ###  Transformation: quantile
    pipe_xgb_quantile = Pipeline([
                ('model', TransformedTargetRegressor(regressor=XGBRegressor(),
                    transformer=QuantileTransformer(n_quantiles=100, output_distribution="normal"))
                )])

    ###  Transformation: Box-cox 
    pipe_xgb_boxcox = Pipeline([
                ('model', TransformedTargetRegressor(regressor=XGBRegressor(),
                    transformer=PowerTransformer(method="box-cox", standardize=False) # def=False:
                    #func=np.reciprocal, inverse_func=np.expm1
                    )
                )])

    ###  Transformation: square root 
    pipe_xgb_sqrt = Pipeline([
        ('model', TransformedTargetRegressor(regressor=XGBRegressor(),
            func=np.sqrt, inverse_func=np.square #np.expm1
        )
        )])

    joblib.dump(pipe_en, './pipelines/pipe_en.pkl')
    joblib.dump(pipe_bag_en, './pipelines/pipe_bag_en.pkl')
    joblib.dump(pipe_en_log, './pipelines/pipe_en_log.pkl')
    joblib.dump(pipe_en_quantile, './pipelines/pipe_en_quantile.pkl')
    joblib.dump(pipe_en_boxcox, './pipelines/pipe_en_boxcox.pkl')
    joblib.dump(pipe_en_sqrt, './pipelines/pipe_en_sqrt.pkl')

    joblib.dump(pipe_xgb, './pipelines/pipe_xgb.pkl')
    joblib.dump(pipe_xgb_log, './pipelines/pipe_xgb_log.pkl')
    joblib.dump(pipe_xgb_quantile, './pipelines/pipe_xgb_quantile.pkl')
    joblib.dump(pipe_xgb_boxcox, './pipelines/pipe_xgb_boxcox.pkl')
    joblib.dump(pipe_xgb_sqrt, './pipelines/pipe_xgb_sqrt.pkl')



if __name__ == "__main__":
    main()
