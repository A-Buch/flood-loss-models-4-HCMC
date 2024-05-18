#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pipelines for feature selection"""

__author__ = "Anna Buch, Heidelberg University"
__email__ = "a.buch@stud.uni-heidelberg.de"


import joblib
from pathlib import Path


from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

import settings as s

seed = s.seed

OUTPATH_PIPES = Path(s.OUTPATH_PIPES)

def main():
    ############ Logistic Regression ##########
    pipe_logreg = Pipeline(
        steps=[
            ("scaler", MinMaxScaler()),
            ("model", LogisticRegression(random_state=seed)),
            # , class_weight="balanced"))
        ]
    )

    ############ Random Forest Classifier ##########

    pipe_rf = Pipeline(
        steps=[
            ("scaler", MinMaxScaler()),
            (
                "model",
                RandomForestClassifier(
                    random_state=seed,
                    # class_weight="balanced"
                    class_weight={0: 0.60, 1: 0.40},  # 0.4 0.6
                ),
            ),
        ]
    )

    ############ Random Forest Regressor - REFERENCE MODEL ##########

    pipe_ref_model = Pipeline(
        steps=[
            ("scaler", MinMaxScaler()),
            (
                "model",
                RandomForestRegressor(
                    random_state=seed,
                ),
            ),
        ]
    )

    ############  Elastic Net  ##################
    pipe_en = Pipeline(
        steps=[
            ("scaler", MinMaxScaler()),
            ("model", ElasticNet(random_state=seed)),
        ]
    )
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
        "model": BaggingRegressor,  # default bootstrap=True
        "kwargs": {
            "estimator": ElasticNet(random_state=seed),
            "bootstrap": True,
            #'random_state':seed
        },  # TODO: pass 'random_state':seed to baggingregressor
    }
    pipe_en_bag = Pipeline([("scaler", MinMaxScaler()), ("bagging", ensemble_model["model"](**ensemble_model["kwargs"]))])

    ############  XGBoost Regressor  ##################

    pipe_xgb = Pipeline(
        steps=[
            ("scaler", MinMaxScaler()),
            ("model", XGBRegressor(random_state=seed)),
        ]
    )

    ###########  Conditional Random Forest ##############
    pipe_crf = "cforest"  ## "pipe_crf" specifies CRF model from Rpackage

    ## pkl file for models

    # REFERENCE model
    joblib.dump(pipe_ref_model, OUTPATH_PIPES / "pipe_ref_model.pkl")

    # CLASSIFICATION models
    joblib.dump(pipe_rf, OUTPATH_PIPES / "pipe_rf.pkl")
    joblib.dump(pipe_logreg, OUTPATH_PIPES / "pipe_logreg.pkl")

    # REGRESSION models
    joblib.dump(pipe_en, OUTPATH_PIPES / "pipe_en.pkl")
    joblib.dump(pipe_en_bag, OUTPATH_PIPES / "pipe_en_bag.pkl")
    joblib.dump(pipe_xgb, OUTPATH_PIPES / "pipe_xgb.pkl")
    joblib.dump(pipe_crf, OUTPATH_PIPES / "pipe_crf.pkl")



if __name__ == "__main__":
    main()
    