#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data preprocessing for HCMC survey dataset"""

__author__ = "Anna Buch, Heidelberg University"
__email__ = "a.buch@stud.uni-heidelberg.de"

# ## Feature selection 
# Enitre workflow with all models for the target variables relative content loss and business reduction (degree of loss) as well for the binary version of relative content loss (chance of loss)
# 
# Due to the samll sample size a nested CV is used to have the possibility to even get generalization error, in the inner CV the best hyperaparamters based on k-fold are selected; in the outer cv the generalization error across all tested models is evaluated. A seprate unseen validation set as done by train-test split would have an insufficent small sample size.
# Nested CV is computationally intensive but with the samll sample size and a well chosen set of only most important hyperparameters this can be overcome.
# 
# - Logistic Regression (binary rcloss)
# - Elastic Net
# - eXtreme Gradient Boosting
# - Random Forest
# 

import sys, os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import itertools

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import make_scorer, mean_absolute_error, mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns

#sys.path.append(os.getcwd()+ '../../')
sys.path.insert(0, "../../")
import utils.feature_selection as fs
import utils.training as t
import utils.evaluation as e
import utils.evaluation_metrics as em
import utils.figures as f
import utils.settings as s
import utils.pipelines as p
import utils.preprocessing as pp

p.main()  # create/update model settings
#s.init()
seed = s.seed

pd.set_option('display.max_columns', None)
plt.figure(figsize=(20, 10))

import contextlib
import warnings
warnings.filterwarnings('ignore')

#### Load R packages to process Conditional Random Forest in python
# *Note 1: all needed R packages have to be previously loaded in R*
# *Note 2: Make sure that caret package version >= 6.0-81, otherwise caret.train() throws an error*
import rpy2
from rpy2.robjects import pandas2ri, Formula
from rpy2.robjects.packages import importr, data

# get basic R packages
utils = importr('utils')
base = importr('base')
dplyr = importr('dplyr')
stats_r = importr("stats")  # rename due to similar python package

# pandas.DataFrames to R dataframes 
pandas2ri.activate()

# print r df in html
import rpy2.ipython.html
rpy2.ipython.html.init_printing()

# get libraries for CRF processing, ctree_controls etc
#partykit = importr('partykit') # for single Conditional Inference tree
party = importr('party')        # Random Forest with Conditional Inference Trees (Conditional Random Forest)
permimp = importr('permimp')  # conditional permutation feature importance
caret = importr('caret') # package version needs to be higher than  >=  6.0-90
nestedcv = importr('nestedcv')
tdr = importr("tdr")


targets = ["Target_relative_contentloss_euro", "Target_businessreduction"]
target = targets[0]

## settings for cv
kfolds_and_repeats = 3, 1  # <k-folds, repeats> for nested cv
cv = RepeatedKFold(n_splits=kfolds_and_repeats[0], n_repeats=kfolds_and_repeats[1], random_state=seed)

## save models and their evaluation in following folders:
Path(f"../models_trained/degree_of_loss/nested_cv_models").mkdir(parents=True, exist_ok=True)
Path(f"../models_trained/degree_of_loss/final_models").mkdir(parents=True, exist_ok=True)
Path(f"../models_evaluation/degree_of_loss").mkdir(parents=True, exist_ok=True)
Path(f"../selected_features/degree_of_loss").mkdir(parents=True, exist_ok=True)


df_candidates = pd.read_excel("../../input_survey_data/input_data_contentloss_tueb.xlsx")
#df_candidates = pd.read_excel("../../input_survey_data/input_data_businessreduction_tueb.xlsx")

## test drop flow velocity due to diffenret flooding sources (eg. overwhelmed draingage systems)
df_candidates = df_candidates.drop("flowvelocity", axis=1)


print(df_candidates.shape)
df_candidates.tail(2)

## Fit model 
score_metrics = {
    "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
    "RMSE": make_scorer(em.root_mean_squared_error, greater_is_better=False),
    "MBE": make_scorer(em.mean_bias_error, greater_is_better=False),
    "R2": "r2",
    "SMAPE": make_scorer(em.symmetric_mean_absolute_percentage_error, greater_is_better=False)
}



## empty variables to store model outputs
eval_sets = {}
models_trained = {}
final_models_trained = {}
models_coef = {}
predicted_values = {}
df_feature_importances = pd.DataFrame(index=df_candidates.drop(target, axis=1).columns.to_list())
models_scores = {}

## iterate over piplines. Each pipline contains a scaler and regressor (and optionally a bagging method) 
pipelines = ["pipe_crf", "pipe_en", "pipe_xgb"]  
# pipelines = ["pipe_en"]  

## Load set of hyperparamters
hyperparams_set = pp.load_config("../../utils/hyperparameter_sets.json")


for pipe_name in pipelines:

    model_name = pipe_name.split('_')[1]
    print( f"\n############ Applying {model_name} on {target} ############\n ")

    df_Xy = df_candidates
    X_names = df_Xy.drop(target, axis=1).columns.to_list()

    ## remove zero-loss records only for combined dataset
    if target == "Target_relative_contentloss_euro":
        print(f"Removing {df_Xy.loc[df_Xy[target]==0.0,:].shape[0]} zero loss records")
        df_Xy = df_Xy.loc[df_Xy[target]!=0.0,:]
        print(f"Keeping {df_Xy.shape} damage cases (excluding zero-loss cases) for model training and evaluation")


    ## drop samples where target is nan
    print(f"Dropping {df_Xy[f'{target}'].isna().sum()} records from entire dataset due that these values are nan in target variable")
    df_Xy = df_Xy[ ~df_Xy[f"{target}"].isna()]

    ## Elastic Net and Random Forest: drop samples where any value is nan
    if (model_name == "en") | (model_name == "crf"):
        df_Xy.dropna(inplace=True)

    print(
        "Using ",
        df_Xy.shape[0],
        " records, from those are ",
        (df_Xy[target][df_Xy[target] == 0.0]).count(),
        " cases with zero-loss or zero-reduction",
    )

    X = df_Xy[X_names]
    y = df_Xy[target]

    ## load model pipelines and hyperparameter space
    pipe = joblib.load(f'./pipelines/{pipe_name}.pkl')
    param_space = hyperparams_set[f"{model_name}_hyperparameters"]

    ## if bagging is used
    if "bag" in pipe_name.split("_"):
        print(f"Testing {model_name} with bagging")
        param_space = { k.replace('model', 'bagging__estimator') : v for (k, v) in param_space.items()}


    if model_name != "crf":

        ## fit model for unbiased model evaluation and for final model used for Feature importance, Partial Dependence etc.
        mf = t.ModelFitting(
            model=pipe, 
            Xy=df_Xy,
            target_name=target,
            param_space=hyperparams_set[f"{model_name}_hyperparameters"],
            tuning_score="neg_mean_absolute_error",
            cv=cv,
            kfolds_and_repeats=kfolds_and_repeats,
            seed=seed,
        )
        models_trained_ncv = mf.model_fit_ncv()

        # save models from nested cv and final model on entire ds
        joblib.dump(models_trained_ncv, f"../models_trained/degree_of_loss/nested_cv_models/{model_name}_{target}.joblib")
            
        ## evaluate model    
        me = e.ModelEvaluation(
            models_trained_ncv=models_trained_ncv, 
            Xy=df_Xy,
            target_name=target,
            score_metrics=score_metrics,
            cv=cv,
            kfolds=kfolds_and_repeats[0],
            seed=seed,
        )
        model_evaluation_results = me.model_evaluate_ncv()

        
        ## visual check if hyperparameter ranges are good or need to be adapted
        for i in range(len(model_evaluation_results["estimator"])):
            print(f"{model_name}: ", model_evaluation_results["estimator"][i].best_params_)


        ## store models evaluation 
        models_scores[model_name] =  {
            k: model_evaluation_results[k] for k in tuple("test_" + s for s in list(score_metrics.keys()))
        } # get evaluation scores, metric names start with "test_<metricname>"


        ## Final model

        ## get final model based on best MAE score during outer cv
        best_idx = list(models_scores[model_name]["test_MAE"]).index(max(models_scores[model_name]["test_MAE"]))
        final_model = model_evaluation_results["estimator"][best_idx]
        print("used params for best model:", final_model.best_params_)  # use last model as the best one
        final_model = final_model.best_estimator_

        ## predict on entire dataset and save final model
        y_pred = final_model.predict(X) 
        final_models_trained[model_name] = final_model 
        joblib.dump(final_model, f"../models_trained/degree_of_loss/final_models/{model_name}_{target}.joblib")


        ## Feature importance of best model
        importances = me.permutation_feature_importance(final_model, repeats=5)


        ## regression coefficients for linear models
        with contextlib.suppress(Exception): 
            # models_coef[model_name] = me.calc_regression_coefficients(final_model)
            # outfile = f"../models_evaluation/regression_coefficients_{model_name}_{target}.xlsx"
            # models_coef[model_name].round(3).to_excel(outfile, index=True)
            # print("Regression Coefficients:\n", models_coef[model_name].sort_values("probabilities", ascending=False), f"\n.. saved to {outfile}")

            ## try to get coefficients of final predictor
            # models_coef[model_name] = me.calc_regression_coefficients(final_model, sanity_test=True)
            import statsmodels.api as sm
            from scipy import stats
            from sklearn.linear_model import LinearRegression

            def calc_standard_error(y, y_pred, newX):  # TODO move them outside class or to utils.py
                MSE = (sum((y - y_pred)**2))/(len(newX)-len(newX[0]))
                ## MSE = (sum((y-y_pred)**2))/(len(newX)-len(X.columns))
                var_b = MSE*(np.linalg.inv(np.dot(newX.T, newX)).diagonal())
                return np.sqrt(var_b)
            def calc_p_values(ts_b, newX):
                p_values =  [2*(1-stats.t.cdf(np.abs(i),(len(newX)-len(newX[0])))) for i in ts_b]
                return p_values

            ## make sanity check
            sanity_test = True
            if sanity_test:
                X_exog = MinMaxScaler().fit_transform(X)#, 
                # y = self.y

                ## reference: p-values from statsmodels
                m = sm.OLS(y, sm.add_constant(X_exog))
                m_res = m.fit()
                #print(m_res.summary())
                p_values_reference = m_res.summary2().tables[1]['P>|t|']

                ## self calculated p-values
                reg = LinearRegression().fit(X_exog, y)
                y_pred_test = reg.predict(X_exog)
                coefs_intercept = np.append(reg.intercept_, list(reg.coef_))

                ## calc p-values
                newX = np.append(np.ones((len(X_exog),1)), X_exog, axis=1)
                sd_b = calc_standard_error(y, y_pred_test, newX)  # standard error calculated based on MSE of newX
                ts_b = coefs_intercept / sd_b        # t values
                p_values = calc_p_values(ts_b, newX)   # significance

                assert (list(np.round(p_values_reference, 3)) == np.round(p_values, 3)).all(), sys.exit("different calculation of p values")

            ## get coefficients and intercept
            model_coefs = final_model.named_steps['model'].coef_
            model_intercept = final_model.named_steps['model'].intercept_
            coefs_intercept = np.append(model_intercept, list(model_coefs))
            
            ## calc significance of coefficient,  modified based on : https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
            ## calc p-values
            ## FIXME errorneous calculation of p-values when moved inside the class
            newX = np.append(np.ones((len(X),1)), X, axis=1)
            sd_b = calc_standard_error(y, y_pred, newX)  # standard error calculated based on MSE of newX
            ts_b = coefs_intercept / sd_b        # t values
            p_values = calc_p_values(ts_b, newX)   # significance

            model_coef = pd.DataFrame(
                {
                    "features": ["intercept"] + X.columns.to_list(),
                    "coefficients": np.round(coefs_intercept, 4),
                    "standard errors": np.round(sd_b, 3),
                    "t values": np.round(ts_b, 3),
                    "probabilities": np.round(p_values, 5),
                }, index=range(len(coefs_intercept))
            )
            models_coef[model_name] = model_coef
            # outfile = f"../models_evaluation/degree_of_loss/regression_coefficients_{model_name}_{target}_{year}_{aoi_and_floodtype}.xlsx"
            # models_coef[model_name].round(3).to_excel(outfile, index=True)
            print("Regression Coefficients:\n", models_coef[model_name].sort_values("probabilities", ascending=False))
    
    else:
        ## normalize X 
        ## not mandatory for CRF but solves bug in party.cforest() and potentially decreases processing time
        # X_crf = pd.DataFrame(MinMaxScaler().fit_transform(X),   
        #     columns=df_Xy.loc[:, df_Xy.columns!=target].columns
        # )    
        # # ## save evaluation set for later usage in feature importance
        # eval_sets[f"{model_name}"] = pd.concat([df_Xy[target], X_crf], axis=1) 

        ## define model settings
        mf = t.ModelFitting(
            model="cforest",  # name of applied R algorithm 
            Xy=df_Xy,
            target_name=target,
            param_space=hyperparams_set[f"{model_name}_hyperparameters"],
            tuning_score="neg_mean_absolute_error",
            cv=cv,
            kfolds_and_repeats=kfolds_and_repeats,
            seed=s.seed
        )
        models_trained_ncv = mf.r_model_fit_ncv()  # pipe
        final_model = mf.r_final_model_fit()
        
        me = e.ModelEvaluation(
            models_trained_ncv=models_trained_ncv, 
            Xy=df_Xy,
            target_name=target,
            score_metrics=score_metrics,  # make optional in ModelEvlaution() class
            cv=cv,
            kfolds=kfolds_and_repeats[0],
            seed=s.seed
        )
        model_evaluation_results = me.r_model_evaluate_ncv()

        ## get std of CRF from inner folds
        ## TODO shorter name for model_evaluation_results_dict
        model_evaluation_results_dict =  {a : [] for a in ["test_MAE", "test_RMSE", "test_MBE", "test_R2", "test_SMAPE"]}
        for idx in range(1, kfolds_and_repeats[0]+1):  # number of estimators , R counts starting from 1
            df = me.r_models_cv_predictions(idx)  # get all crf estimators from inner cv
            model_evaluation_results_dict['test_MAE'].append(mean_absolute_error(df.testy, df.predy))
            model_evaluation_results_dict['test_RMSE'].append(em.root_mean_squared_error(df.testy,df.predy)) #(df.testy, df.predy)
            model_evaluation_results_dict['test_MBE'].append(em.mean_bias_error(df.testy, df.predy))
            model_evaluation_results_dict['test_R2'].append(em.r2_score(df.testy, df.predy))
            model_evaluation_results_dict['test_SMAPE'].append(em.symmetric_mean_absolute_percentage_error(df.testy, df.predy))

        ## Feature importance of best model
        importances = me.r_permutation_feature_importance(final_model)

        ## store model evaluation
        models_scores[model_name] = model_evaluation_results_dict ## store performance scores from R estimators
        final_models_trained[model_name] = final_model


    # ## Evaluation

    ## store fitted models and their evaluation results for later 
    eval_sets[model_name] = df_Xy
    models_trained[f"{model_name}"] = models_trained_ncv
    predicted_values[model_name] = me.residuals

    
    print("\nSelect features based on permutation feature importance")
    df_importance = pd.DataFrame(
        {
            f"{model_name}_importances" : importances[0],   # averaged importnace scores across repeats
            f"{model_name}_importances_std" : importances[1]
        },
        index=X_names,
    )
    df_feature_importances = df_feature_importances.merge(
        df_importance[f"{model_name}_importances"],   # only use mean FI, drop std of FI
        left_index=True, right_index=True, how="outer")
    print("5 most important features:", df_feature_importances.iloc[:5].index.to_list())
            


## Print model evaluation based on performance on outer cross-validation 
## TODO remove overhead
xgb_model_evaluation = pd.DataFrame(models_scores["xgb"]).mean(axis=0)  # get mean of outer cv metrics (negative MAE and neg RMSE, pos. R2, pos MBE, posSMAPE)
xgb_model_evaluation_std = pd.DataFrame(models_scores["xgb"]).std(axis=0)   # get respective standard deviations
crf__model_evaluation = pd.DataFrame(models_scores["crf"]).mean(axis=0)
crf_model_evaluation_std = pd.DataFrame(models_scores["crf"]).std(axis=0)
en_model_evaluation = pd.DataFrame(models_scores["en"]).mean(axis=0)
en_model_evaluation_std = pd.DataFrame(models_scores["en"]).std(axis=0)

model_evaluation = pd.concat([en_model_evaluation, en_model_evaluation_std, xgb_model_evaluation, xgb_model_evaluation_std, crf__model_evaluation, crf_model_evaluation_std], axis=1)
model_evaluation.columns = ["en_score", "en_score_std", "xgb_score", "xgb_score_std", "crf_score", "crf_score_std"]

model_evaluation.index = model_evaluation.index.str.replace("test_", "")
model_evaluation.loc["MAE"] = model_evaluation.loc["MAE"].abs()
model_evaluation.loc["RMSE"] = model_evaluation.loc["RMSE"].abs()

outfile = f"../models_evaluation/degree_of_loss/performance_{target}.xlsx"
model_evaluation.round(3).to_excel(outfile, index=True)
print("Outer evaluation scores:\n", model_evaluation.round(3), f"\n.. saved to {outfile}")


## Feature Importances 

#### prepare Feature Importances 
## Have the same feature importance method across all applied ML models
## Weight Importances by model performance on outer loop (mean MAE)
## **Overall FI ranking (procedure similar to Rözer et al 2019; Brill 2022)**

## weight FI scores based on performance ; weigth importances from better performed models stronger
model_weights =  {
    "xgb_importances" : np.abs(models_scores["xgb"]["test_MAE"].mean()),
    "en_importances" : np.abs(models_scores["en"]["test_MAE"].mean()),
    "crf_importances" : np.mean(np.abs(models_scores["crf"]["test_MAE"])),
}

df_feature_importances_w = fs.calc_weighted_sum_feature_importances(df_feature_importances, model_weights)


####  Plot Feature importances

## the best model has the highest weighted feature importance value
# df_feature_importances_w.describe()

df_feature_importances_plot = df_feature_importances_w

## drop features which dont reduce the loss
#df_feature_importances_plot = df_feature_importances_plot.loc[df_feature_importances_plot.weighted_sum_importances > 2, : ] 

f.plot_stacked_feature_importances(
    df_feature_importances_plot[["crf_importances_weighted", "en_importances_weighted", "xgb_importances_weighted",]],
    target_name=target,
    model_names_plot = ("Conditional Random Forest", "Elastic Net", "XGBoost"),
    outfile=f"../models_evaluation/degree_of_loss/feature_importances_{target}.jpg"
)


### Save final feature space 
## The final selection of features is used later for the non-parametric Bayesian Network

## drop records with missing target values
print(f"Dropping {df_candidates[f'{target}'].isna().sum()} records from entire dataset due that these values are nan in target variable")
df_candidates = df_candidates[ ~df_candidates[target].isna()]
print(f"Keeping {df_candidates.shape[0]} records and {df_candidates.shape[1]} features")


## sort features by their overall importance (weighted sum across across all features) 
final_feature_names = df_feature_importances_w["weighted_sum_importances"].sort_values(ascending=False).index##[:10]
print(final_feature_names)

## save importnat features, first column contains target variable
fs.save_selected_features(
    df_candidates, 
    pd.DataFrame(df_candidates, columns=[target]), 
    final_feature_names,
    filename=f"../selected_features/degree_of_loss/final_predictors_{target}.xlsx"
)
