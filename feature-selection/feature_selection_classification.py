#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data preprocessing for HCMC survey dataset"""

__author__ = "Anna Buch, Heidelberg University"
__email__ = "a.buch@stud.uni-heidelberg.de"

## Feature selection for content losses done by Logistic Regression

# Due to many zero losses especially in content losses, a binary regression was tested to distinguish between occured losses and no losses. 
# The before applied elastic net result showed that the elastic net algorithm might be a bit too complex for the moderate size of training set 
# and the imbalaanced distribution with in the response (many zero losses compared to only a very a left skewed distribution of occured content losses)  
# *Sources*
# Geron 2019: https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch04.html#idm45022190228392


import os, sys
from pathlib import Path
import joblib
import re
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, precision_score, make_scorer

import matplotlib.pyplot as plt


UTILS_PATH = os.path.join(os.path.abspath(''), '../', 'utils')
sys.path.append(UTILS_PATH)
import training as t
import evaluation as e
import evaluation_utils as eu
import settings as s
import preprocessing as pp

seed = s.seed

import contextlib
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
plt.figure(figsize=(20, 10))

## settings for cv
kfolds_and_repeats = 10, 5 # 10, 3 # 10, 5  # <k-folds, repeats> for nested cv
inner_cv = RepeatedStratifiedKFold(n_splits=kfolds_and_repeats[0], n_repeats=kfolds_and_repeats[1], random_state=seed)
outer_cv = RepeatedStratifiedKFold(n_splits=kfolds_and_repeats[0], n_repeats=1, random_state=seed)


## save models and their evaluation in following folders:
INPATH_DATA = Path(s.INPATH_DATA) # input path
OUTPATH_FEATURES, OUTPATH_FINALMODELS, OUTPATH_ESTIMATORS_NCV, OUTPATH_RESULTS = [ # create output paths
    pp.create_output_dir(Path(d) / "chance_of_rcloss") for d in  
    [s.OUTPATH_FEATURES, s.OUTPATH_FINALMODELS, s.OUTPATH_ESTIMATORS_NCV, s.OUTPATH_EVAL]
]
print(OUTPATH_FEATURES, OUTPATH_FINALMODELS, OUTPATH_ESTIMATORS_NCV, OUTPATH_RESULTS)


targets = [("chance of rcloss", "chance of rcloss")]
target, target_plot = targets[0]
pred_target = f"pred_{target}"

# Get logger  # test: init application
main_logger = "__feature_extraction_chance__"
logger = s.init_logger(main_logger)


## load DS for relative content loss
df_candidates = pd.read_excel(f"{INPATH_DATA}/input_data_contentloss_tueb.xlsx")
# change target name for component for rclsos "degree of rcloss" in  s.feature_names_plot 
s.feature_names_plot["Target_relative_contentloss_euro"]  = "chance of rcloss"
##  use nice feature names
df_candidates.rename(columns=s.feature_names_plot, inplace=True)

## test drop flow velocity due to diffenret flooding sources (eg. overwhelmed draingage systems)
# df_candidates = df_candidates.drop("flowvelocity", axis=1)
logger.info("Variables from input DS", df_candidates.describe())


# Variables for average classification report  # TODO move to utils functions
originalclass = []
predictedclass = []
def custom_scorer(y_true, y_pred):
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)
    return precision_score(y_true, y_pred) # return precision score
    #return accuracy_score(y_true, y_pred)

## Evaluation metrics
score_metrics = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1_macro": "f1_macro", # only for class balanced
    # "f1": "f1",
}

## Load set of hyperparamters
hyperparams_set = pp.load_config(f"{UTILS_PATH}/hyperparameter_sets.json")

## iterate over piplines. Each pipline contains a scaler and regressor (and optionally a bagging method) 
pipelines = ["pipe_logreg"]
# pipelines = ["pipe_sgd"] # with logreg loss and en penality


## empty variables to store model outputs
eval_sets = {}
models_trained = {}
final_models_trained = {}
models_coef = {}
predicted_values = {}
df_feature_importances = pd.DataFrame(index=df_candidates.drop(target, axis=1).columns.to_list())
models_scores = {}

for pipe_name in pipelines:

    TIME0 = datetime.now()

    ## load model pipeline and get model name
    pipe = joblib.load(f"{UTILS_PATH}/pipelines/{pipe_name}.pkl")


    try:
        model_name = re.findall("[a-zA-Z]+", str(pipe.steps[1][1].__class__).split(".")[-1])[0] # get model name for python models  
    except AttributeError:
        model_name = pipe # get R model name
    

    print( f"\n  ############### Applying {model_name} on {target}:  ###############\n")

    ## load hyperparameter space
    param_space = hyperparams_set[f"{model_name}_hyperparameters"]

    ## if bagging is used, adapt hyperparameeter names
    if "bag" in pipe_name.split("_"):
        print(f"Testing {model_name} with bagging")
        param_space = { k.replace('model', 'bagging__estimator') : v for (k, v) in param_space.items()}


    ## load input dataset
    df_Xy = df_candidates

    # rm geometry column which only needed for visualization
    df_Xy = df_Xy.drop("geometry", axis=1)

    
    ## drop content value var due its only needed to recalculate losses after BN
    with contextlib.suppress(Exception):
        df_Xy.drop(["shp_business_limitation"], axis=1, inplace=True)
        df_Xy.drop(["closs"], axis=1, inplace=True)
        df_Xy.drop(["shp_content_value_euro"], axis=1, inplace=True)
        df_Xy.drop(["shp_sector"], axis=1, inplace=True)
        
        
    ## get predictor names
    X_names = df_Xy.drop(target, axis=1).columns.to_list()

 
    #save to find out which samples are predicted wrongly
    df_candidates_continous = df_candidates.copy()
    df_candidates_continous.dropna(inplace=True)
    
    # ## test impact of median imputation on model performance
    # print("test impact of median imputation on model performance")
    # df_Xy[X_names] = df_Xy[X_names].apply(lambda x: x.fillna(x.median()),axis=0)
    df_Xy.dropna(inplace=True)
    print("Drop records with missing values",
        f"keeping {df_Xy.shape} damage cases for model training and evaluation")
   

    ## set target as binary class
    df_Xy[target][df_Xy[target] > 0] = 1
    df_Xy[target] = df_Xy[target].astype(float)#("Int64")

    ## clean df from remaining records containg nan
    df_Xy.dropna(inplace=True)

    print("Amount of missing target values should be zero: ", df_Xy[target].isna().sum())
    print(
        "Using ",
        df_Xy.shape[0],
        " records, from those are ",
        (df_Xy[target][df_Xy[target] == 0.0]).count(),
        " cases with zero-loss or zero-reduction",
    )

    X = df_Xy[X_names]
    y = df_Xy[target]
    print(y.describe())
    print("Used predictors:", X.columns)
    print("Response variable:", y.name)


    if model_name != "crf":

        ## save evaluation set for later usage in feature importance
        eval_sets[f"{model_name}"] = df_Xy 
        mf = t.ModelFitting(
            model=pipe, 
            Xy=df_Xy,
            target_name=target,
            param_space=param_space,
            tuning_score="f1_macro",#"precision", #"f1",#"f1_macro", #"roc_auc", #"f1_macro", #"accuracy", # best 54%="precision",#", #"f1_macro", #"precision", #"f1_macro", # accuracy
            cv=inner_cv,
            kfolds_and_repeats=kfolds_and_repeats,
            seed=seed,
        )
        models_trained_ncv = mf.model_fit_ncv()  # pipe
        
        # from sklearn.utils.class_weight import compute_sample_weight
        # sample_weights = np.where(df_candidates_continous[target].between(0.01, 0.05), 0.5, 0.5) 
        # sample_weights = np.where(df_candidates_continous[target].between(0.01, 0.05), 0.8, 0.2)  ## NOTE curr best number of y_pred with 72 damage cases (0.6/0.4 weigth=15 damage cases in y_pred )
        sample_weights = np.where(df_candidates_continous[target] > 0.01, 0.6, 0.4) # org
        # sample_weights = np.where(df_candidates_continous[target].between(0.0001, 0.05), 0.4, 0.6)  # 0..6 / 0.4 give ca 56 damage cases in ypred NOTE 0.8 / 0.2 44 damage cases in ypred
        # no or only slight sample weight, too high weight such as .8, 0.2 = too less non-damage cases are predicted in NCV
        print(pd.Series(sample_weights).describe())
        print(pd.Series(sample_weights).value_counts())
        
           
        me = e.ModelEvaluation(
            models_trained_ncv=models_trained_ncv, 
            Xy=df_Xy,
            target_name=target,
            score_metrics=score_metrics,
            cv=outer_cv,
            kfolds=kfolds_and_repeats[0],
            seed=seed,
        )
        #model_evaluation_results = me.model_evaluate_ncv(prediction_method="predict_proba")
        model_evaluation_results = me.model_evaluate_ncv(
            sample_weights=None, 
            # sample_weights={"model__sample_weight": sample_weights}, 
            prediction_method="predict_proba"
            )


        ## Classification report for nested cross validation                
        nested_score = cross_val_score(models_trained_ncv, X=X, y=y, cv=outer_cv, 
               scoring=make_scorer(custom_scorer))
        # Average values in classification report for all folds in a K-fold Cross-validation  
        # print(nested_score) # scores from each fold
        print("classification report from outer cross validation") 
        print(classification_report(originalclass, predictedclass)) 


        ## visual check if hyperparameter ranges are good or need to be adapted
        for i in range(len(model_evaluation_results["estimator"])):
            print(f"{model_name}: ", model_evaluation_results["estimator"][i].best_params_)


        ## store models evaluation 
        models_scores[model_name] =  {
            k: model_evaluation_results[k] for k in tuple("test_" + s for s in list(score_metrics.keys()))
        } # get evaluation scores, metric names start with "test_<metricname>"

        
        ## Final model

        ## get final model based on best MAE score during outer cv
        best_idx = list(models_scores[model_name]["test_f1_macro"]).index(max(models_scores[model_name]["test_f1_macro"]))
        final_model = model_evaluation_results["estimator"][best_idx]
        print("used params for best model:", final_model.best_params_)
        final_model = final_model.best_estimator_

        ## get predictions of final model from respective outer test set
        test_set_best = df_Xy.iloc[model_evaluation_results["indices"]["test"][best_idx], :]
        finalmodel_X_test = test_set_best.drop(target, axis=1)
        finalmodel_y_pred_proba = final_model.predict_proba(test_set_best.drop(target, axis=1))  # get predictions from final model for its test-set (should be the same as done during model evluation with ncv)
        finalmodel_y_true = test_set_best[target]

        ## store highest predicted probabilities and respective predictions
        finalmodel_y_pred = np.argmax(finalmodel_y_pred_proba, axis=1)  # predicted class (non-damage-case, damage-case)
        finalmodel_y_proba = np.take_along_axis(  # predicted probability that a case is non-damage or damage-case
            finalmodel_y_pred_proba, 
            np.expand_dims(finalmodel_y_pred, axis=1), 
            axis=1
        )
        finalmodel_y_proba = finalmodel_y_proba.flatten()
        final_models_trained[model_name] = final_model 
        joblib.dump(final_model, f"{OUTPATH_FINALMODELS}/{model_name}_{target}.joblib")

        ## Feature importance of best model
        importances = me.permutation_feature_importance(  # acces predction error via MAE cirterion
            final_model, 
            finalmodel_X_test, finalmodel_y_pred, 
            repeats=5)

        ## regression coefficients for linear models
        with contextlib.suppress(Exception):
            models_coef[model_name] = me.calc_regression_coefficients(final_model, finalmodel_y_true, finalmodel_y_pred)
            outfile = f"{OUTPATH_RESULTS}/regression_coefficients_{model_name}_{target}.xlsx"
            models_coef[model_name].round(3).to_excel(outfile, index=True)
            print("Regression Coefficients:\n", models_coef[model_name].sort_values("probabilities", ascending=False), f"\n.. saved to {outfile}")
            
            ## check if any regression coefficient is significant 
            if np.min(models_coef[model_name]["probabilities"]) >= 0.05:
                ## manually decorate init_logger, extending with creation of log file for warnings
                logger = s.decorate_init_logger(s.init_logger)("__warning_coefs__") 
                logger.warning("non of the regression coefficients is significant")


    eval_sets[model_name] = df_Xy
    models_trained[f"{model_name}"] = models_trained_ncv

    ## store and save model predictions from all outer test sets
    predicted_values[model_name] = me.residuals  # y_train, y_pred and residual from nested cv
    predicted_values[model_name]["y_true_rcloss"] = df_candidates_continous[target]

    ## NOTE preserve index from loaded input dataset, 
    ## use predicted probabilities of loss as input for the estimation of relative content loss
    print(predicted_values[model_name].describe())
    predicted_values[model_name].to_excel(f"{OUTPATH_FEATURES}/predictions_{target.replace(' ', '_')}.xlsx", index=True)

    ## Feature importance
    print("\nSelect features based on permutation feature importance")
    df_importance = pd.DataFrame(
        {
            f"{model_name}_importances" : importances[0],   # mean importnaces across repeats
            f"{model_name}_importances_std" : importances[1]
        },
        index=X_names,
    )
    df_feature_importances = df_feature_importances.merge(
        df_importance[f"{model_name}_importances"],   # only use mean FI, drop std of FI
        left_index=True, right_index=True, how="outer")

    df_feature_importances = df_feature_importances.sort_values(f"{model_name}_importances", ascending=False)  # get most important features to the top
    print("5 most important features:", df_feature_importances.iloc[:5].index.to_list())
    #df_importance = df_importance.loc[df_importance[f"{model_name}_importances"] >= 0.000000, : ]

    print(
    f"\nTraining and evaluation of {model_name} took {(datetime.now() - TIME0).total_seconds()} seconds\n"
    )


## Print model evaluation based on performance on outer cross-validation 
classifier_model_evaluation = pd.DataFrame(models_scores[model_name]).mean(axis=0)  # get mean of outer cv metrics (negative MAE and neg RMSE, pos. R2, pos MBE, posSMAPE)
classifier_model_evaluation_std = pd.DataFrame(models_scores[model_name]).std(axis=0)   # get respective standard deviations

model_evaluation = pd.concat([classifier_model_evaluation, classifier_model_evaluation_std], axis=1)
model_evaluation.columns = [f"{model_name}_score", f"{model_name}_score_std"]

model_evaluation.index = model_evaluation.index.str.replace("test_", "")

outfile = f"{OUTPATH_RESULTS}/performance_{target}.xlsx"
model_evaluation.round(3).to_excel(outfile, index=True)
print("Outer evaluation scores:\n", model_evaluation.round(4), f"\n.. saved to {outfile}")

## Print evaluation nested cv
print("y true: \n", predicted_values[model_name]["y_true"].value_counts())
print("y pred from nested cv: \n", pd.Series(predicted_values[model_name]["y_pred"]).value_counts())

# ### Empirical median ~ predicted median
for k,v in predicted_values.items():
    print(f"\n{k} estimators from nested cross-validation:")
    print(eu.empirical_vs_predicted(predicted_values[k]["y_true"], predicted_values[k]["y_pred"]))
