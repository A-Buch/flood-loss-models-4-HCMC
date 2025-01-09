#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions for model fitting"""

__author__ = "Anna Buch, Heidelberg University"
__email__ = "anna.buch@uni-heidelberg.de"

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV

import settings as s

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

seed = s.seed
logger = s.init_logger("__model_training__")


base = importr("base")
pdp = importr("pdp")
nestedcv = importr("nestedcv")
# partykit = importr('partykit') # for single Conditional Inference tree
party = importr("party")  # Random Forest with Conditional Inference Trees (Conditional Random Forest)
caret = importr("caret")  # package version needs to be higher than  >=  6.0-90


class ModelFitting(object):
    """
    sklearn models and R model training by nested cross-validation
    """

    def __init__(self, model, Xy, target_name, param_space, tuning_score, cv, kfolds_and_repeats: tuple, seed):
        # super(model_fitting, self).__init__()  # super() == to call parent class

        ## properties
        self.model = model  # algorithm for sklearn model
        self.final_model = None
        self.r_algorithm_name: str = str(model)  # name of algorithm for R model  ## TODO move non-global properies to methods()
        self.X = pd.DataFrame(Xy.drop(target_name, axis=1))
        self.X = pd.DataFrame(MinMaxScaler().fit_transform(self.X), columns=self.X.columns)
        self.y: pd.DataFrame = Xy[target_name]
        self.target_name: str = target_name
        self.param_space: dict = param_space
        self.tuning_score: str = tuning_score
        self.k_folds, self.repeats = kfolds_and_repeats
        self.inner_cv = cv
        self.outer_cv = cv
        self.seed: int = seed

    def r_tunegrid(self, mtry_min, mtry_max, mtry_seq):
        """
        Expand grid for Hyperparameter tuning for Conditonal Random Forest, only hyperparameter "mtry" can be tuned
        mtry_min, mtry_max, mtry_seq (int): parameter space for mtry, mtry_max should not be larger than number of predictors
        return: Function for tuning cforest models
        """
        robjects.r("""
            r_tunegrid <- function(mtry_min, mtry_max, mtry_seq, verbose=FALSE) {
                expand.grid(mtry = seq(mtry_min, mtry_max, mtry_seq))
            }""")
        r_tunegrid = robjects.globalenv["r_tunegrid"]
        return r_tunegrid(mtry_min, mtry_max, mtry_seq)

    def model_fit_ncv(self):
        """
        Optimazation of sklearn model by nested cross-validation [inner folds]
        return: k-best models of inner folds
        """
        ## define inner cv, model training with hyperparameter tuning
        models_trained_ncv = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.param_space,
            cv=self.inner_cv,
            scoring=self.tuning_score,
            refit=True,
            random_state=self.seed,
        )
        return models_trained_ncv
        # return super().model_fit_ncv(**kwargs)

    def r_model_fit_ncv(self):
        """
        Optimatzation of R model by nested cross-validation  [inner and outer folds]
        return: k-best models of outer folds
        """
        ## define inner cv, model training with hyperparameter tuning
        base.set_seed(seed)
        models_trained_ncv = nestedcv.nestcv_train(
            y=self.y,
            x=self.X,
            method=self.r_algorithm_name,
            outer_train_predict=True,  # save predictions on outer folds
            n_outer_folds=self.k_folds,
            n_inner_folds=self.k_folds,
            finalCV=True,  ## model is CV fitted again on entire dataset to determine final model
            tuneGrid=self.r_tunegrid(self.param_space["mtry"][0], self.param_space["mtry"][1], self.param_space["mtry"][2]),
            # pass_outer_folds=True,  # =False - otherwise repeations of inner Folds not possible, =True -use same outer fold to build finalModel
            # =True find final model based on CV on outer folds, same folds and no repeats used for model evaluation
            savePredictions="final",  # mandatory to call to get final model (best estimator from outer folds) via $final_fit
            metric="MAE",
            controls=party.cforest_unbiased(ntree=300),
            trControl=caret.trainControl(
                method="repeatedcv",
                number=self.k_folds,
                repeats=self.repeats,
                savePredictions="all",  # "final"= save predictions for the optimal tunning params
                search="random",
            ),
        )
        logger.info(f"\nSummary CRF \n {base.summary(models_trained_ncv)}")

        return models_trained_ncv

    def r_final_model(self):
        """
        Get final R model, which is the best-performed estimator (smallest MAE) from outer folds
        return: best-performed R model
        """
        robjects.r("""
            r_final_model <- function(model, verbose=FALSE) {
                model$final_fit$finalModel
            }
        """)
        final_model = robjects.globalenv["r_final_model"]
        final_model = final_model(self.r_model_fit_ncv())

        return self.final_model
