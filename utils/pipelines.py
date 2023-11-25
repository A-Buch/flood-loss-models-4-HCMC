#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pipelines for feature selection"""

__author__ = "Anna Buch, Heidelberg University"
__email__ = "a.buch@stud.uni-heidelberg.de"


import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingRegressor, BaggingClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet, SGDClassifier
from xgboost import XGBRegressor


from sklearn.pipeline import Pipeline

import utils.settings as s


s.init()
seed = s.seed

def main():


    ############ Logistic Regression ##########

    pipe_logreg = Pipeline( steps = [
        ('scaler', MinMaxScaler()), 
        ('model', LogisticRegression(random_state=seed)) 
    ] )

    # Logistic Regression with Bagging
    ensemble_model = {
        'model': BaggingClassifier,
        #'kwargs': {'estimator': LogisticRegression()},
        'kwargs': {'estimator': LogisticRegression(class_weight={0:0.40, 1:0.60})},
        }
    pipe_logreg_bag = Pipeline([
        ('scaler', MinMaxScaler()),
        ('bagging', ensemble_model['model'] (**ensemble_model['kwargs']) )
    ])

    ############  Stochastic Gradient Decent classifier  ##################
    pipe_sgd = Pipeline( steps = [
        ('scaler', MinMaxScaler()), 
        ('model', SGDClassifier(loss='log_loss', random_state=seed)) # log_loss’ gives logistic regression, a probabilistic classifier.
    ] )

              
    ############  Elastic Net  ##################
    pipe_en = Pipeline( steps = [
            ('scaler', MinMaxScaler()), 
            ('model', ElasticNet(random_state=seed)),
    ])
    #  Transformation: Box-cox 
    # pipe_en = Pipeline([
            # ('scaler', MinMaxScaler()), 
            # ('model', TransformedTargetRegressor(regressor=ElasticNet(),
                # transformer=PowerTransformer(method="box-cox", standardize=False) # def=False:
                # #func=np.reciprocal, inverse_func=np.expm1
            # )
                # )])


    # Elastic Net with Bagging
    ensemble_model = {
        'model': BaggingRegressor,   # default bootstrap=True
        'kwargs': {'estimator': ElasticNet(random_state=seed),
                   'bootstrap': True,
                   #'random_state':seed
                  }  # TODO: pass 'random_state':seed to baggingregressor
    }
    pipe_en_bag = Pipeline([
        ('scaler', MinMaxScaler()),
        ('bagging', ensemble_model['model'] (**ensemble_model['kwargs']) )
    ])


    ############  XGBoost Regressor  ##################

    pipe_xgb = Pipeline(steps = [
        ('scaler', MinMaxScaler()), 
        #('model', SelectFromModel(
        #    XGBRegressor(random_state=seed),
        #    max_features=10,
        #)
        ('model', XGBRegressor(random_state=seed)),
    ])


    ###########  Conditional Random Forest ##############
    pipe_crf = "cforest"
    ## --> it seems to be possible to incoporate R models into sklearn pipeline but for this use case this implementation is out of scope
    ## R model is called directly in python scripts


    joblib.dump(pipe_logreg, './pipelines/pipe_logreg.pkl')
    joblib.dump(pipe_logreg_bag, './pipelines/pipe_logreg_bag.pkl')
    
    joblib.dump(pipe_sgd, './pipelines/pipe_sgd.pkl')

    joblib.dump(pipe_en, './pipelines/pipe_en.pkl')
    joblib.dump(pipe_en_bag, './pipelines/pipe_en_bag.pkl')

    joblib.dump(pipe_xgb, './pipelines/pipe_xgb.pkl')

    joblib.dump(pipe_crf, './pipelines/pipe_crf.pkl')


if __name__ == "__main__":
    main()
