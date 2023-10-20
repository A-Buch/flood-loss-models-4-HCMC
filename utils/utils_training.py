#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions for model fitting"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold

import utils_feature_selection as fs
import utils.utils_evaluation as e
import utils.settings as s

#me = e.model_evaluation()
#s.init()
seed = s.seed

# load r library initally
#%load_ext rpy2.ipython

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri # pandas.DataFrames to R dataframes 

pandas2ri.activate()
rpy2.ipython.html.init_printing()

# get basic R packages
base = importr('base')

# pandas.DataFrames to R dataframes 
from rpy2.robjects import pandas2ri, Formula
pandas2ri.activate()

# print r df in html
import rpy2.ipython.html
rpy2.ipython.html.init_printing()


pdp = importr("pdp")
nestedcv = importr("nestedcv")
# get libraries for CRF processing, ctree_controls etc
#partykit = importr('partykit') # for single Conditional Inference tree
party = importr('party')        # Random Forest with Conditional Inference Trees (Conditional Random Forest)
caret = importr('caret') # package version needs to be higher than  >=  6.0-90



class ModelFitting(object):

    def __init__(self, model, Xy, target_name, param_space, seed):
        #super(model_fitting, self).__init__()  # super() == to call parent class
        ## properties
        self.model = model   
        self.X = pd.DataFrame(Xy.drop(target_name, axis=1))
        self.X = pd.DataFrame(
                MinMaxScaler().fit_transform(self.X),   
                columns=self.X.columns) 
        self.y: pd.DataFrame = Xy[target_name]
        self.target_name = target_name
        self.param_space = param_space
        self.k_folds: int = 5
        self.inner_cv = RepeatedKFold(n_splits=self.k_folds, n_repeats=10, random_state=seed)
        self.outer_cv = RepeatedKFold(n_splits=self.k_folds, n_repeats=10, random_state=seed)
        self.seed: int = seed
    
    

    def r_tunegrid(self, mtry_min, mtry_max, mtry_seq):
        """
        Expand grid for Hyperparameter tuning for Conditonal Random Forest, only hyperparameter "mtry" can be tuned
        mtry_min, mtry_max, mtry_seq (int): parameter space for mtry, mtry_max should not be larger than number of predictors
        return: Function for tuning cforest models
        """
        robjects.r('''
            r_tunegrid <- function(mtry_min, mtry_max, mtry_seq, verbose=FALSE) {
                expand.grid(mtry = seq(mtry_min, mtry_max, mtry_seq))
            }''')
        r_tunegrid = robjects.globalenv['r_tunegrid']
        return r_tunegrid(mtry_min, mtry_max, mtry_seq)

    def model_fit_ncv(self): #, **kwargs):
        """
        """
        # sourcery skip: inline-immediately-returned-variable
    #def model_fit(self, **kwargs):
        #model = self.model#kwargs["model"]

        ## define inner cv, model training with hyperparameter tuning
        models_trained_ncv = RandomizedSearchCV( # GridSearchCV( 
            estimator=self.model ,
            param_distributions=self.param_space,
            cv=self.inner_cv, 
            scoring="neg_mean_absolute_error",
            refit=True,   
            random_state=self.seed,
        )
        return models_trained_ncv
        #return super().model_fit_ncv(**kwargs)
    
    def final_model_fit(self):
        """
        Train final model on entire dataset to get Variable Selection
        return: best-performed sklearn model
        """
        final_model = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.param_space,
            cv=self.outer_cv, 
            scoring="neg_mean_absolute_error",
            refit=True,   
            random_state=self.seed,
        )
        final_model.fit(self.X, self.y)
        
        return final_model.best_estimator_


    # ## decorator for R model
    # def decorator_func(self, model, Xy, target_name, param_space, seed, feature_name, scale=True):
    #     """
    #     Decorator to get partial dependence instead of python-sklearn-model from R-party-model
    #     """
    def r_model_fit_ncv(self):   # fun = call function which should be decorate
        """
        """    
            # def wrapper(*args, **kwargs):  
                ### no need to call original function due that it is entirely replaced by decorator
                #x = func()

        # ## define inner cv, model training with hyperparameter tuning
        base.set_seed(seed)
        models_trained_ncv = nestedcv.nestcv_train(
            y=self.y, 
            x=self.X,    # tree-based models uses information gain / gini coefficient inherently which will not be affected by scaling 
                            # normalize X --> not mandatory for CRF but solves bug in party.cforest() and potentially decreases processing time
            method="cforest",
            #savePredictions="final", # predictions on inner folds
            outer_train_predict=True, # predictions on outer training fold
            n_outer_folds=self.k_folds,
            n_inner_folds=self.k_folds,
            finalCV=False,  # "NA" final model fitting is skipped altogether, which gives a useful speed boost if performance metrics are all that is needed.
            tuneGrid=e.r_tunegrid(
            self.param_space["mtry"][0], 
            self.param_space["mtry"][1], 
            self.param_space["mtry"][2]
        ), 
        #filterFUN = ttest_filter, filter_options = list(nfilter = 300),
            metric='MAE',#'RMSE',  # RMSE unit of target or use MAE due that more robust than RMSE further metrics options Rsquared, RMSE, MAE 
                # RMSE penalizes large gaps more harshly than MAE
            controls = party.cforest_unbiased(
                ntree = 300,  # didnt improved with 200 or 500 trees
                # mincriterion = 0.05,   # the value of the test statistic (for testtype == "Teststatistic"), or 1 - p-value (for other values of testtype) that must be exceeded in order to implement a split.
            ),  
            trControl = caret.trainControl(
                method = "repeatedcv",  # "oob" - then no repeats are needed
                number = self.k_folds,   ## = number of splits
                repeats = self.k_folds,  #  nbr repeats == number of tried values for mtry
                savePredictions = "final"  # saves predictions from optimal tuning parameters
            )
        )
        print("\nSummary CRF \n", base.summary(models_trained_ncv))
        
        return models_trained_ncv
            

    def r_final_model_fit(self, models_trained_ncv):
        # sourcery skip: inline-immediately-returned-variable
        """
        """
        best_hyperparameters = fs.r_dataframe_to_pandas(fs.r_best_hyperparamters(models_trained_ncv))
        final_model = party.cforest(Formula(f'{self.target_name} ~ .'),  
            data=pd.concat([self.y.reset_index(), self.X],   # use normalized X
                    axis=1,
                ).drop("index", axis=1),
            control= party.cforest_unbiased(mtry=best_hyperparameters.mtry, ntree=300)
        )
        return final_model

    # return r_model_fit_ncv, r_final_model_fit
