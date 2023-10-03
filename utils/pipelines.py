#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pipelines for feature selection"""

import joblib
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingRegressor, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor

from sklearn.pipeline import Pipeline
# from imblearn.pipeline import Pipeline
#from sklearn.preprocessing import QuantileTransformer, quantile_transform, PowerTransformer, power_transform
#rom sklearn.compose import TransformedTargetRegressor

import utils.settings as s


s.init()
seed = s.seed

def main():


    ############ Logistic Regression ##########

    pipe_logreg = Pipeline( steps = [('scaler', MinMaxScaler()), ('model', LogisticRegression(random_state=seed)) ] )

    # Logistic Regression with Bagging   #TODO check if LogReg with Bagging really makes sense? 
    ensemble_model = {
        'model': BaggingClassifier,   # default bootstrap=True
        'kwargs': {'estimator': LogisticRegression(random_state=seed),  # estimator -> variable from Bagging Regressor
                   #'bootstrap': True,
                   #'random_state':seed
                  }#,  # TODO: pass 'random_state':seed to baggingregressor
         #'parameters':
    }
    pipe_logreg_bag = Pipeline([
        ('scaler', MinMaxScaler()),
        ('bagging', ensemble_model['model'] (**ensemble_model['kwargs']) )
    ])


    ############  Elastic Net  ##################

    pipe_en = Pipeline( steps = [('scaler', MinMaxScaler()), ('model', ElasticNet(random_state=seed)) ] )

    # Elastic Net with Bagging
    ensemble_model = {
        'model': BaggingRegressor,   # default bootstrap=True
        'kwargs': {'estimator': ElasticNet(random_state=seed),  # estimator -> variable from Bagging Regressor
                   'bootstrap': True,
                   #'random_state':seed
                  }  # TODO: pass 'random_state':seed to baggingregressor
    }
    pipe_en_bag = Pipeline([
        ('scaler', MinMaxScaler()),
        ('bagging', ensemble_model['model'] (**ensemble_model['kwargs']) )
    ])


    ############  XGBoost Regressor  ##################

    pipe_xgb = Pipeline( steps = [('scaler', MinMaxScaler()), ('model', XGBRegressor(random_state=seed)) ] )


    ###########  Conditional Random Forest ##############
    pipe_crf = None
    ## TODO try if its possible to add R-based model settings into python Pipeline()


    joblib.dump(pipe_logreg, './pipelines/pipe_logreg.pkl')
    joblib.dump(pipe_logreg_bag, './pipelines/pipe_logreg_bag.pkl')

    joblib.dump(pipe_en, './pipelines/pipe_en.pkl')
    joblib.dump(pipe_en_bag, './pipelines/pipe_en_bag.pkl')

    joblib.dump(pipe_xgb, './pipelines/pipe_xgb.pkl')

    joblib.dump(pipe_crf, './pipelines/pipe_crf.pkl')


if __name__ == "__main__":
    main()
