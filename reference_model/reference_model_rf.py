#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reference model"""

__author__ = "Anna Buch, Heidelberg University"
__email__ = "anna.buch@uni-heidelberg.de"

# Reference model to compare with BN results

"""
NOTE:
- reference mode to predict rcloss and bred
- use reference model to verify BN performances for HCMC ds and CanTho Dataset
- for rcloss: train with zero-loss cases, aim is to compare applied approach (chance-of-loss + degree-of-loss + BN)
- always basic set of hyperparameters is used (no tuning),
- evaluate with 10 fold cross-valdiation
"""

import os
from datetime import datetime
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import make_scorer, mean_absolute_error

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import src.training as t
import src.evaluation as e
import src.evaluation_utils as eu
import src.figures as f
import src.settings as s
import src.preprocessing as pp

import contextlib
import warnings


SRC_PATH = os.path.join(os.path.abspath(""), "./", "src")

seed = s.seed

pd.set_option("display.max_columns", None)
plt.figure(figsize=(20, 10))


warnings.filterwarnings("ignore")


## settings for cv
kfolds_and_repeats = 5, 5  # 10, 5 # 3, 1  # <k-folds, repeats> for nested cv
inner_cv = RepeatedKFold(n_splits=kfolds_and_repeats[0], n_repeats=kfolds_and_repeats[1], random_state=seed)
outer_cv = RepeatedKFold(n_splits=kfolds_and_repeats[0], n_repeats=1, random_state=seed)  # make same as for R nestedcv.train()


targets = [
    ("rcloss", "relative content loss", ("darkgrey", "steelblue"), ("teal", "firebrick")),
    ("rbred", "relative business reduction", ("lightsteelblue", "steelblue"), ("teal", "firebrick", "indianred")),
]

target, target_plot, hue_colors, plot_cdf_colors = targets[0]  # <-- change here target variable
# target, target_plot, hue_colors, plot_cdf_colors = targets[1]   # <-- change here target variable

## test RF on degree of rcloss in HCMC and Can Tho <-> compare with transferbaility of BN for degree of rcloss
test_degree_of_rcloss = False  # False = default :apply models on rcloss; True : models on degree of rcloss


# Get and init logger
main_logger = f"__reference_model_{target}__"
logger = s.init_logger(main_logger)


## save models and their evaluation in following folders:
INPATH_DATA = Path(s.INPATH_DATA)  # input path
OUTPATH_FEATURES, OUTPATH_FINALMODELS, OUTPATH_ESTIMATORS_NCV, OUTPATH_RESULTS = [  # create output paths
    pp.create_output_dir(Path(re.sub(r"(model)", r"reference_\1", d)) / target)  # adapt filepaths with prefix and addtional AOI folder
    # pp.create_output_dir(Path(re.sub(r"(model_)", rf"reference_\1", d)) / aoi / target)  # adapt filepaths with prefix and addtional AOI folder
    for d in [s.OUTPATH_FEATURES, s.OUTPATH_FINALMODELS, s.OUTPATH_ESTIMATORS_NCV, s.OUTPATH_EVAL]
]


## load candidate predictors and target
df_candidates = pd.read_excel(f"{INPATH_DATA}/input_data_contentloss_tueb.xlsx")
# df_candidates = pd.read_excel("{INPATH_DATA}/input_data_businessreduction_tueb.xlsx")


##  use nice feature names
df_candidates.rename(columns=s.feature_names_plot, inplace=True)


logger.info(df_candidates.shape)

## Evaluation metrics
score_metrics = {
    "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
    "RMSE": make_scorer(eu.root_mean_squared_error, greater_is_better=False),
    "MBE": make_scorer(eu.mean_bias_error, greater_is_better=False),
    # "R2": "r2",
    "SMAPE": make_scorer(eu.symmetric_mean_absolute_percentage_error, greater_is_better=False),
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
pipelines = ["pipe_ref_model"]

## Load set of hyperparamters
hyperparams_set = pp.load_config(f"{SRC_PATH}/hyperparameter_sets.json")


for pipe_name in pipelines:
    TIME0 = datetime.now()

    ## load model pipelines
    pipe = joblib.load(f"{SRC_PATH}/pipelines/{pipe_name}.pkl")

    try:
        model_name = re.findall("[a-zA-Z]+", str(pipe.steps[1][1].__class__).split(".")[-1])[0]  # get model name for python models
    except AttributeError:
        model_name = pipe  # get R model name

    ## load respective hyperparameter space
    param_space = hyperparams_set[f"{model_name}_hyperparameters"]

    ## if bagging fro model training is used , rename hyperparmeters
    if "bag" in pipe_name.split("_"):
        logger.info(f"Testing {model_name} with bagging")
        param_space = {k.replace("model", "bagging__estimator"): v for (k, v) in param_space.items()}

    logger.info(f"\n############ Applying {model_name} on {target} ############\n ")

    # save original df for later
    df_Xy = df_candidates

    # rm geometry column which only needed for visualization

    with contextlib.suppress(Exception):
        df_Xy = df_Xy.drop("geometry", axis=1)
        if target == "rcloss":
            df_Xy.drop("shp_content_value_euro", axis=1, inplace=True)
            df_Xy.drop("closs", axis=1, inplace=True)
            df_Xy.drop("shp_sector", axis=1, inplace=True)
            df_Xy.drop("shp_business_limitation", axis=1, inplace=True)
        if target == "rbred":
            df_Xy.drop("shp_business_limitation", axis=1, inplace=True)
            df_Xy.drop("shp_content_value_euro", axis=1, inplace=True)
            df_Xy.drop("shp_sector", axis=1, inplace=True)

            # df_candidates.drop("hh_monthly_income_euro", axis=1, inplace=True)

    if test_degree_of_rcloss:
        ## remove zero-loss records only for rcloss
        if target == "rcloss":
            logger.info(f"Removing {df_Xy.loc[df_Xy[target]==0.0,:].shape[0]} zero loss records")
            df_Xy = df_Xy.loc[df_Xy[target] != 0.0, :]

    # get predictor names
    X_names = df_Xy.drop(target, axis=1).columns.to_list()
    logger.info(f"Predictor names: {X_names}")

    ## drop samples where target is nan
    logger.info(f"Removing {df_Xy[target].isna().sum()} records from entire dataset due that these values are nan in target variable")
    # df_Xy = df_Xy[ ~df_Xy[target].isna()]

    ## Elastic Net and Random Forest: drop samples where any value is nan
    if (model_name == "RandomForestRegressor") | (model_name == "ElasticNet") | (model_name == "cforest"):
        print("Dropping records with missing values")
        df_Xy.dropna(inplace=True)

    logger.info(
        f"Finally use {df_Xy.shape[0]} records for feature extraction, from those are {(df_Xy[target][df_Xy[target] == 0.0]).count()} cases with zero-loss or zero-reduction",
    )

    X = df_Xy[X_names]
    y = df_Xy[target]

    ## run sklearn model
    # print("value ranges of features:", df_Xy.describe())

    ## fit model for unbiased model evaluation and for final model used for Feature importance, Partial Dependence etc.
    mf = t.ModelFitting(
        model=pipe,
        Xy=df_Xy,
        target_name=target,
        param_space=param_space,
        tuning_score=score_metrics["MAE"],  # tune by getting reducing MAE
        cv=inner_cv,
        kfolds_and_repeats=kfolds_and_repeats,
        seed=seed,
    )
    models_trained_ncv = mf.model_fit_ncv()

    # save models from nested cv and final model on entire ds
    joblib.dump(models_trained_ncv, f"{OUTPATH_ESTIMATORS_NCV}/{model_name}_{target}.joblib")

    ## evaluate model
    me = e.ModelEvaluation(
        models_trained_ncv=models_trained_ncv,
        Xy=df_Xy,
        target_name=target,
        score_metrics=score_metrics,
        cv=outer_cv,
        kfolds=kfolds_and_repeats[0],
        seed=seed,
    )
    model_evaluation_results = me.model_evaluate_ncv()

    ## visual check if hyperparameter ranges are good or need to be adapted
    logger.info("Performance of best estimators on outer test-sets:")
    for i in range(len(model_evaluation_results["estimator"])):
        print(f"{model_name}: ", model_evaluation_results["estimator"][i].best_params_)

    ## store models evaluation
    models_scores[model_name] = {
        k: model_evaluation_results[k] for k in tuple("test_" + s for s in list(score_metrics.keys()))
    }  # get evaluation scores, metric names start with "test_<metricname>"

    ## reverse sklearn.cross_validate() outputted regression scores (e.g. MAE, RMSE, SMAPE, R2)
    models_scores[model_name] = me.negate_scores_from_sklearn_cross_valdiate(models_scores[model_name])

    ## Final model

    ## get final model based on best MAE score during outer cv
    best_idx = list(models_scores[model_name]["test_MAE"]).index(min(models_scores[model_name]["test_MAE"]))
    final_model = model_evaluation_results["estimator"][best_idx]
    logger.info(f"Params of best model: {final_model.best_params_}")
    final_model = final_model.best_estimator_

    logger.info("Performance of best model:")
    for metric in models_scores[model_name].keys():
        print(metric, models_scores[model_name][metric][best_idx])

    ## predict on entire dataset and save final model
    final_models_trained[model_name] = final_model
    joblib.dump(final_model, f"{OUTPATH_FINALMODELS}/{model_name}_{target}.joblib")

    ## get predictions of final model from respective outer test set
    test_set_best = df_Xy.iloc[model_evaluation_results["indices"]["test"][best_idx], :]
    finalmodel_X_test = test_set_best.drop(target, axis=1)
    finalmodel_y_test = test_set_best[target]
    finalmodel_y_pred = final_model.predict(
        finalmodel_X_test
    )  # get predictions from final model for its test-set (should be the same as done during model evluation with ncv)

    ## Learning curve of train and test set of final model
    train_set_best = df_Xy.iloc[model_evaluation_results["indices"]["train"][best_idx], :]
    f.plot_learning_curves(
        final_model, train_set_best, test_set_best, target, f"{OUTPATH_RESULTS}/learning_curves_{target}_{model_name}.png", model_name
    )

    ## Feature importance of best model
    importances = me.permutation_feature_importance(final_model, finalmodel_X_test, finalmodel_y_test, repeats=5)
    ## init df to store FIs
    df_feature_importances = pd.DataFrame(index=df_candidates.drop(target, axis=1).columns.to_list())

    # ## Summarize all models and their evaluation

    ## store fitted models and their evaluation results for later
    eval_sets[model_name] = df_Xy
    models_trained[f"{model_name}"] = models_trained_ncv
    predicted_values[model_name] = me.residuals

    ## store Feature Importances of each model
    logger.info("\nSelect features based on permutation feature importance")
    df_importance = pd.DataFrame(
        {
            f"{model_name}_importances": importances[0],  # averaged importnace scores across repeats
            f"{model_name}_importances_std": importances[1],
        },
        index=X_names,
    )
    df_feature_importances = df_feature_importances.merge(
        df_importance[f"{model_name}_importances"],  # only use mean FI, drop std of FI
        left_index=True,
        right_index=True,
        how="outer",
    )
    df_feature_importances = df_feature_importances.sort_values(
        f"{model_name}_importances", ascending=False
    )  # get most important features to the top
    logger.info(f"5 most important features: {df_feature_importances.iloc[:5].index.to_list()}")

    logger.info(f"\nTraining and evaluation of {model_name} took {(datetime.now() - TIME0).total_seconds() / 60} minutes\n")


## Plot performance ranges of all evaluated estimators from outer cross-validation
logger.info("Creating boxplots for range of performane scores from outer folds of nested cross-validation")
f.boxplot_outer_scores_ncv(models_scores, outfile=f"{OUTPATH_RESULTS}/boxplot_scores4ncv_{target}.png", target_name=target_plot)


# store avergaed scores and std for later usage
model_evaluation = pd.DataFrame(models_scores["RandomForestRegressor"]).mean(
    axis=0
)  # get mean of outer cv metrics (negative MAE and neg RMSE, pos. R2, pos MBE, posSMAPE)
model_evaluation_std = pd.DataFrame(models_scores["RandomForestRegressor"]).std(axis=0)  # get respective standard deviations

model_evaluation = pd.concat([model_evaluation, model_evaluation_std], axis=1)
model_evaluation.columns = ["RandomForestRegressor_score", "RandomForestRegressor_score_std"]

## rename metrics
model_evaluation.index = model_evaluation.index.str.replace("test_", "")

outfile = f"{OUTPATH_RESULTS}/performance_{target}.xlsx"
model_evaluation.round(3).to_excel(outfile, index=True)
# logger.info(f"Outer evaluation scores of nested cross-validation (mean) :\n {model_evaluation.round(3)} \n.. saved to {outfile}")
logger.info(f"Outer evaluation scores of nested cross-validation (median) :\n {model_evaluation.round(3)} \n.. saved to {outfile}")


## Feature Importances
### drop features which dont reduce the loss
df_feature_importances_plot = df_feature_importances
df_feature_importances_plot = df_feature_importances_plot.loc[df_feature_importances_plot[f"{model_name}_importances"] > 0.0, :]
df_feature_importances_plot = df_feature_importances_plot.sort_values(f"{model_name}_importances", ascending=True)


## TODO update with plt_fi() func as soons a its more flexible in number of models passed to func()
sns.set_style("whitegrid", {"axes.grid": False})
plt.figure(figsize=(30, 22), facecolor="w")
fig = df_feature_importances_plot.plot.barh(
    color="darkblue",
    width=0.5,
    alpha=0.7,
)
plt.xlabel("Importance")
plt.ylabel("")
# plt.title(f"Feature Importances for {target.replace('_',' ')}")

top_bar = mpatches.Patch(
    color="darkblue",
    label=f"{model_name.replace('cR','c R')}",
    alpha=0.7,
)
plt.tick_params(axis="x", which="major", labelsize=12)
plt.tick_params(axis="y", which="major", labelsize=12)
plt.legend(handles=[top_bar], loc="lower right")
plt.tight_layout()
plt.grid(False)

fig.get_figure().savefig(f"{OUTPATH_RESULTS}/feature_importances_{target}.png", bbox_inches="tight")
plt.close()

## Save final feature space
### The final selection of features is used later for the non-parametric Bayesian Network

## drop records with missing target values
logger.info(f"Dropping {df_candidates[f'{target}'].isna().sum()} records from entire dataset due that these values are nan in target variable")
df_candidates = df_candidates[~df_candidates[target].isna()]
logger.info(f"Keeping {df_candidates.shape[0]} records and {df_candidates.shape[1]} features")


### Empirical ~ predicted
## use y_pred cross-valdiated from outer folds, mulitplied by 100 for more readable output
for k, v in predicted_values.items():
    logger.info(f"\n{k} observed and predicted target distirbution from cross-valdiated outer folds:")
    logger.info(eu.empirical_vs_predicted(predicted_values[k]["y_true"], predicted_values[k]["y_pred"]))

logger.info("\nObserved and predicted target distirbution from final model:")
logger.info(eu.empirical_vs_predicted(finalmodel_y_test, finalmodel_y_pred))


## of all best-performed estimators
f.plot_observed_predicted(
    # finalmodel_y_test, finalmodel_y_pred,   # final estimator
    me.residuals["y_true"],
    me.residuals["y_pred"],
    hue=None,  # results_p["chance of closs"],
    xlabel=f"Observed {target} [%]",
    ylabel=f"Predicted {target} [%]",
    legend=False,  # True
    outfile=rf"{OUTPATH_RESULTS}/scatter_obs_pred_ref_{target}_hcmc.png",
)


 
# ### Scatterplot


logger.info("creating scatterplotd")

predicted_values[model_name]["dummy_obs_binary"] = predicted_values[model_name]["y_true"]
predicted_values[model_name].loc[predicted_values[model_name]["dummy_obs_binary"] > 0.0, "dummy_obs_binary"] = 1
predicted_values[model_name].describe()



## final estimator
f.plot_observed_predicted(
    # finalmodel_y_test, finalmodel_y_pred,
    me.residuals["y_true"],
    me.residuals["y_pred"],
    hue=predicted_values[model_name]["dummy_obs_binary"],  # results_p["chance of closs"],
    xlabel=f"Observed {target} [%]",
    ylabel=f"Predicted {target} [%]",
    # xlabel="observed rcloss", ylabel="predicted rcloss",
    legend=False,  # True
    outfile=f"{OUTPATH_RESULTS}/scatter_obs_pred_ref_{target}_hcmc.png",
)


 
# ## Transfer and apply best-performed RF model on CanTHo


logger.info("## Transfer and apply best-performed RF model on CanTHo")

## load Can THo DS
infile_name = "input_data_contentloss_cantho"
# infile_name = "input_data_businessreduction_cantho"

df_cantho = pd.read_excel(f"{INPATH_DATA}/{infile_name}.xlsx")

##  use nice feature names
df_cantho.rename(columns=s.feature_names_plot, inplace=True)


## prepare CanTHo for BN - adpat with same column structure as training dataset
df_cantho = df_cantho[df_Xy.columns]
df_cantho = df_cantho.dropna()

if test_degree_of_rcloss:
    ## remove zero-loss records only for rcloss
    if target == "rcloss":
        logger.info(f"Removing {df_cantho.loc[df_cantho[target]==0.0,:].shape[0]} zero loss records")
        df_cantho = df_cantho.loc[df_cantho[target] != 0.0, :]


y_true_cantho = df_cantho[target]
df_cantho_X = df_cantho.drop(target, axis=1)
df_cantho.describe()


y_pred_cantho = final_model.predict(df_cantho_X)
y_pred_cantho.shape


# evaluation metrics regiona ltransfered final RF estimator
MAE = eu.mean_absolute_error(y_true_cantho, y_pred_cantho)
MBE = eu.mean_bias_error(y_true_cantho, y_pred_cantho)
RMSE = eu.root_mean_squared_error(y_true_cantho, y_pred_cantho)
SMAPE = eu.symmetric_mean_absolute_percentage_error(y_true_cantho, y_pred_cantho)


logger.info("Model perfromace on CanTho DS: MAE, MBE, RMSE, SMAPE:", MAE, MBE, RMSE, SMAPE)


logger.info(f"\nObserved and predicted target distribution from final model for CanTho ({target}):")
logger.info(eu.empirical_vs_predicted(y_true_cantho, y_pred_cantho))


 
#


pred_cd_cantho = pd.DataFrame(
    {"y_true": pd.Series(y_true_cantho), "y_pred": pd.Series(y_pred_cantho), "residuals": pd.Series(y_pred_cantho) - pd.Series(y_true_cantho)},
    index=pd.Series(y_true_cantho).index,
)

pred_cd_cantho.reset_index(drop=True, inplace=True)
logger.info("Statistics predictions, Can THo", pred_cd_cantho.describe())

for c in pred_cd_cantho.columns:
    ci = eu.calc_confidence_interval(pred_cd_cantho[c], 0.95)

pred_cd_cantho["y_true_binary"] = 0
idx_damage = np.where(pred_cd_cantho["y_true"] >= 1)[0]
pred_cd_cantho.loc[idx_damage, "y_true_binary"] = 1
pred_cd_cantho.describe()




## final estimator
f.plot_observed_predicted(
    pred_cd_cantho["y_true"],
    pred_cd_cantho["y_pred"],
    hue=pred_cd_cantho["y_true_binary"],
    xlabel=f"Observed {target} [%]",
    ylabel=f"Predicted {target} [%]",
    legend=False,  # True
    outfile=f"{OUTPATH_RESULTS}/scatter_obs_pred_ref_{target}_cantho.png",
)



results_p = df_cantho.copy()
results_p["chance of closs"] = 0  # 0: no loss, 1: los occurrs

results_p["chance of closs"][results_p[target] == 0.0] = 0
results_p["chance of closs"][results_p[target] > 0.0] = 1
results_p["chance of closs"].describe()

 
# Plot CDFs


logger.info("Create cumulative distribution functions (CDFs)")

# pred_cd_cantho["residual"] = me.residuals["y_pred"], me.residuals["y_true"]
pred_cd_cantho["residual"] = pred_cd_cantho["y_pred"] - pred_cd_cantho["y_true"]


me.residuals["residual"].describe()


sns.set(rc={"xtick.bottom": True, "ytick.left": True})
sns.set_style(style="ticks", rc={"axes.grid": False, "grid.linestyle": ":"})

# fig, ax = plt.subplots(1, 1)
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

ax.hist(
    [me.residuals["residual"], pred_cd_cantho["residuals"]],  # , residuals_cantho_sparse["residual"]],  # pred HCMC+CanTHo rbred
    bins=200,
    density=True,  # alias for normalize, last bins equals 1
    cumulative=True,
    histtype='step',
    #histtype="stepfilled",
    alpha=0.4,
    color=("teal", "firebrick"),
    # color=("teal","firebrick", "indianred"),
    label=("CDF HCMC", "CDF Can Tho", "CDF Can Tho - sparse"),
)  # # bins and (lognormal / normal) datasets are pre-defined

ax.grid(which="major", color="grey", linewidth=0.5)
# ax.grid( which='minor', color='grey', linewidth=0.3)
# ax.set_yticks([-1.25, -0.75, -0.25,0.24,0.75,1.25], minor=True)


## plt.xticks(np.arange(-80, 50, 10))
plt.xlim(-100, 50)
plt.xticks(np.arange(-100, 56, 20), minor=False)
plt.xticks(np.arange(-100, 56, 10), minor=True)
plt.yticks(np.arange(0, 1.01, 0.1))
plt.yticks(np.arange(0.0, 1.0, 0.05), minor=True)
# plt.axvspan(-30, 30, facecolor='grey', alpha=0.3)

plt.xlabel(f"prediction bias for {target_plot} ({target}) [%]")
plt.ylabel("Probability")
plt.legend(loc="best")

plt.vlines(x=0, ymin=0, ymax=1.01, colors="black", linestyles="--")

plt.title("Cumulative distribution functions (CDFs) of prediction bias, normalized")
plt.savefig(f"{OUTPATH_RESULTS}/cdf_ref_{target}.png", dpi=300, bbox_inches="tight")


logger.info("Create cumulative distribution functions (CDFs) - zoomed in version")

# pred_cd_cantho["residual"] = me.residuals["y_pred"], me.residuals["y_true"]
pred_cd_cantho["residual"] = pred_cd_cantho["y_pred"] - pred_cd_cantho["y_true"]


me.residuals["residual"].describe()


sns.set(rc={"xtick.bottom": True, "ytick.left": True})
sns.set_style(style="ticks", rc={"axes.grid": False, "grid.linestyle": ":"})

# fig, ax = plt.subplots(1, 1)
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

ax.hist(
    [me.residuals["residual"], pred_cd_cantho["residuals"]],  # , residuals_cantho_sparse["residual"]],  # pred HCMC+CanTHo rbred
    bins=200,
    density=True,  # alias for normalize, last bins equals 1
    cumulative=True,
    # histtype='step',
    histtype="stepfilled",
    alpha=0.4,
    color=plot_cdf_colors,
    label=("CDF HCMC", "CDF Can Tho", "CDF Can Tho - sparse"),
)  # # bins and (lognormal / normal) datasets are pre-defined

ax.grid(which="major", color="grey", linewidth=0.5)
ax.grid(which="minor", color="grey", linewidth=0.3)
# ax.set_yticks([-1.25, -0.75, -0.25,0.24,0.75,1.25], minor=True)

## zoomed in
plt.xlim(-30, 30)
plt.xticks(np.arange(-30, 31, 10))
plt.xticks(np.arange(-30, 31, 5), minor=True)
plt.yticks(np.arange(0, 1.01, 0.1))
plt.yticks(np.arange(0.0, 1.0, 0.05), minor=True)


plt.xlabel(f"prediction bias for {target_plot} ({target}) [%]")
plt.ylabel("Probability")
plt.legend(loc="best")

plt.vlines(x=0, ymin=0, ymax=1.01, colors="black", linestyles="--")

plt.title("Cumulative distribution functions (CDFs) of prediction bias, normalized (zoomed in)")
plt.savefig(f"{OUTPATH_RESULTS}/cdf_zoomed_ref_{target}.png", dpi=300, bbox_inches="tight")



