#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pipelines for feature selection"""

__author__ = "Anna Buch, Heidelberg University"
__email__ = "anna.buch@uni-heidelberg.de"


import joblib
from pathlib import Path


from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import src.settings as s

seed = s.seed
OUTPATH_PIPES = Path(s.OUTPATH_PIPES)


############ Logistic Regression ##########
pipe_logreg = Pipeline(
    steps=[
        ("scaler", MinMaxScaler()),
        ("model", LogisticRegression(random_state=seed)),
        # , class_weight="balanced"))
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


###########  Conditional Random Forest ##############
pipe_crf = "cforest"  ## "pipe_crf" specifies CRF model from Rpackage

## pkl file for models

# REFERENCE model
joblib.dump(pipe_ref_model, OUTPATH_PIPES / "pipe_ref_model.pkl")

# CLASSIFICATION models
joblib.dump(pipe_logreg, OUTPATH_PIPES / "pipe_logreg.pkl")

# REGRESSION models
joblib.dump(pipe_crf, OUTPATH_PIPES / "pipe_crf.pkl")
