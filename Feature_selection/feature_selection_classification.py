#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data preprocessing for HCMC survey dataset"""

__author__ = "Anna Buch, Heidelberg University"
__email__ = "a.buch@stud.uni-heidelberg.de"

## Feature selection for content losses done by Logistic Regression

# Due to many zero losses especially in content losses, a binary regression was tested to distinguish between occured losses and no losses. 
# The before applied elastic net result showed that the elastic net algorithm might be a bit too complex for the moderate size of training set 
# and the imbalnced distribution with in the response (many zero losses compared to only a very a left skewed distribution of occured content losses)  
# *Sources*
# Geron 2019: https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch04.html#idm45022190228392


import sys
from pathlib import Path
import numpy as np
import pandas as pd

import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


sys.path.insert(0, "../../")
import utils.feature_selection as fs
import utils.training as t
import utils.evaluation as e
import evaluation_utils as eu
import utils.figures as f
import utils.settings as s
import utils.pipelines as p
import utils.preprocessing as pp

p.main()  # create/update model settings
#s.init()
seed = s.seed

import contextlib
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
plt.figure(figsize=(20, 10))


target = "Target_relative_contentloss_euro"


## settings for cv
kfolds_and_repeats = 10, 5  # <k-folds, repeats> for nested cv   <--- user input
cv = RepeatedStratifiedKFold(n_splits=kfolds_and_repeats[0], n_repeats=kfolds_and_repeats[1], random_state=seed)

## save models and their evaluation in following folders:
Path("../models_trained/chance_of_loss/nested_cv_models").mkdir(parents=True, exist_ok=True)
Path("../models_trained/chance_of_loss/final_models").mkdir(parents=True, exist_ok=True)
Path("../models_evaluation/chance_of_loss").mkdir(parents=True, exist_ok=True)
Path("../selected_features/chance_of_loss").mkdir(parents=True, exist_ok=True)



## load DS for relative content loss
df_candidates = pd.read_excel("../../input_survey_data/input_data_contentloss_tueb.xlsx")
print(df_candidates.shape)

## delete features with more than 10% missing values
# print("Percentage of missing values per feature [%]\n", round(df_candidates.isna().mean().sort_values(ascending=False)[:15]  * 100), 2) 


## Fit model

score_metrics = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1_macro": "f1_macro",
}

## test drop flow velocity due to diffenret flooding sources (eg. overwhelmed draingage systems)
# df_candidates = df_candidates.drop("flowvelocity", axis=1)
df_candidates = df_candidates.drop(["shp_sector_1.0", "shp_sector_2.0","shp_sector_3.0"], axis=1)
print(df_candidates.columns)

## iterate over piplines. Each pipline contains a scaler and regressor (and optionally a bagging method) 

# pipelines = ["pipe_logreg_bag"]
pipelines = ["pipe_logreg"]


eval_sets = {}
models_trained = {}
final_models_trained = {}
models_coef = {}
predicted_values = {}
df_feature_importances = pd.DataFrame(index=df_candidates.drop(target, axis=1).columns.to_list())
models_scores = {}

## Load set of hyperparamters
hyperparams_set = pp.load_config("../../utils/hyperparameter_sets.json")


for pipe_name in pipelines:

    # model_name = re.findall("[a-zA-Z]+", str(pipe.steps[1][1].__class__).split(".")[-1])[0] # works only for python models TODO get this for R model, 
    model_name = pipe_name.split('_')[1]
    print( f"\nApplying {model_name} on {target}:")

    df_Xy = df_candidates
    X_names = df_Xy.drop(target, axis=1).columns.to_list()

    ## test impact of median imputation on model performance
    print("test impact of median imputation on model performance")
    df_Xy[X_names] = df_Xy[X_names].apply(lambda x: x.fillna(x.median()),axis=0)


    ## set target as binary class
    df_Xy[target][df_Xy[target] > 0] = 1
    df_Xy[target] = df_Xy[target].astype("Int64")

    ## clean df from remaining records containg nan
    df_Xy.dropna(inplace=True) ## TODO test with only nan in target removed

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


    ## load model pipelines and hyperparameter space
    pipe = joblib.load(f'./pipelines/{pipe_name}.pkl')
    param_space = hyperparams_set[f"{model_name}_hyperparameters"]

    ## if bagging is used, adapt hyperparameeter names
    if "bag" in pipe_name.split("_"):
        print(f"Testing {model_name} with bagging")
        param_space = { k.replace('model', 'bagging__estimator') : v for (k, v) in param_space.items()}


    if model_name != "crf":

        # ## save evaluation set for later usage in feature importance
        # eval_sets[f"{model_name}"] = df_Xy #pd.concat([df_Xy[target], X_crf], axis=1)         
        mf = t.ModelFitting(
            model=pipe, 
            Xy=df_Xy,
            target_name=target,
            param_space=param_space,
            tuning_score="accuracy",
            cv=cv,
            kfolds_and_repeats=kfolds_and_repeats,
            seed=s.seed,
        )
        models_trained_ncv = mf.model_fit_ncv()  # pipe
        
        me = e.ModelEvaluation(
            models_trained_ncv=models_trained_ncv, 
            Xy=df_Xy,
            target_name=target,
            score_metrics=score_metrics,
            cv=cv,
            kfolds=kfolds_and_repeats[0],
            seed=s.seed
        )
        model_evaluation_results = me.model_evaluate_ncv(prediction_method="predict_proba")


        ## visual check if hyperparameter ranges are good or need to be adapted
        for i in range(len(model_evaluation_results["estimator"])):
            print(f"{model_name}: ", model_evaluation_results["estimator"][i].best_params_)


        ## store models evaluation 
        models_scores[model_name] =  {
            k: model_evaluation_results[k] for k in tuple("test_" + s for s in list(score_metrics.keys()))
        } # get evaluation scores, metric names start with "test_<metricname>"


        ## Final model

        ## get final model based on best MAE score during outer cv
        best_idx = list(models_scores[model_name]["test_accuracy"]).index(max(models_scores[model_name]["test_accuracy"]))
        final_model = model_evaluation_results["estimator"][best_idx]
        print("used params for best model:", final_model.best_params_)  # use last model as the best one
        final_model = final_model.best_estimator_

        ## predict on entire dataset and save final model
        y_pred = final_model.predict(X) 
        final_models_trained[model_name] = final_model 
        joblib.dump(final_model, f"../models_trained/chance_of_loss/final_models/{model_name}_{target}.joblib")

        ## Feature importance of best model
        importances = me.permutation_feature_importance(final_model, repeats=5)

        ## regression coefficients for linear models
        with contextlib.suppress(Exception):   # <-- better than: try and bare except
            # models_coef[model_name] = me.calc_regression_coefficients(final_model)
            # outfile = f"../models_evaluation/chance_of_loss/regression_coefficients_{model_name}_{target}.xlsx"
            # models_coef[model_name].round(3).to_excel(outfile, index=True)
            # print("Regression Coefficients:\n", models_coef[model_name].sort_values("probabilities", ascending=False), f"\n.. saved to {outfile}")

            ## try to get coefficients of final predictor
            # models_coef[model_name] = me.calc_regression_coefficients(final_model, sanity_test=True)

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
            ## FIXME errorneous calculation of p-values when moved inside the class, but works fine when directly applied here
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
            outfile = f"../models_evaluation/chance_of_loss/regression_coefficients_{model_name}_{target}.xlsx"
            models_coef[model_name].round(3).to_excel(outfile, index=True)
            print("Regression Coefficients:\n", models_coef[model_name].sort_values("probabilities", ascending=False))


    # filename = f'./models_trained_ncv/{model_name}_{target}.sav'
    # pickle.dump(model, open(filename, 'wb'))
    eval_sets[model_name] = df_Xy
    models_trained[f"{model_name}"] = models_trained_ncv
    # models_trained[model_name] = model_evaluation_results["estimator"]
    predicted_values[model_name] = me.residuals

    ## Feature importance
    print("\nSelect features based on permutation feature importance")
    df_importance = pd.DataFrame(
        {
            f"{model_name}_importances" : importances[0],   # mean importances across repeats
            f"{model_name}_importances_std" : importances[1]
        },
        index=X_names,
    )
    df_feature_importances = df_feature_importances.merge(
        df_importance[f"{model_name}_importances"],   # only use mean FI, drop std of FI
        left_index=True, right_index=True, how="outer")

    df_feature_importances = df_feature_importances.sort_values(f"{model_name}_importances", ascending=False)  # get most important features to the top
    print("5 most important features:", df_feature_importances.iloc[:5].index.to_list())


## Print model evaluation based on performance on outer cross-validation 
classifier_model_evaluation = pd.DataFrame(models_scores[model_name]).mean(axis=0)  # get mean of outer cv metrics (negative MAE and neg RMSE, pos. R2, pos MBE, posSMAPE)
classifier_model_evaluation_std = pd.DataFrame(models_scores[model_name]).std(axis=0)   # get respective standard deviations

model_evaluation = pd.concat([classifier_model_evaluation, classifier_model_evaluation_std], axis=1)
model_evaluation.columns = [f"{model_name}_score", f"{model_name}_score_std"]

model_evaluation.index = model_evaluation.index.str.replace("test_", "")

outfile = f"../models_evaluation/chance_of_loss/performance_{target}.xlsx"
model_evaluation.round(3).to_excel(outfile, index=True)
print("Outer evaluation scores:\n", model_evaluation.round(3), f"\n.. saved to {outfile}")


## Feature Importances 
### drop features which dont reduce the loss
df_feature_importances_plot = df_feature_importances
df_feature_importances_plot = df_feature_importances_plot.loc[df_feature_importances_plot[f"{model_name}_importances"] >= 0.00, : ] 
df_feature_importances_plot = df_feature_importances_plot.sort_values(f"{model_name}_importances", ascending=True)


## TODO update with plot_stacked_feature_importances() func as soon as it's more flexible in number of models passed to plot_stacked_feature_importances()
plt.figure(figsize=(30, 22), facecolor="w")
fig = df_feature_importances_plot.plot.barh(
    color="darkblue",
    width=0.5,
    )
plt.xlabel("Importance")
plt.ylabel("")
plt.title(f"Feature Importances for {target.replace('_',' ')}")

top_bar = mpatches.Patch(color="darkblue", label=f"{model_name} clasification")
plt.tick_params(axis='x', which='major', labelsize=12)
plt.tick_params(axis='y', which='major', labelsize=12)
plt.legend(handles=[top_bar], loc="lower right")
plt.tight_layout()
plt.grid(False)
plt.show()

 
fig.get_figure().savefig(f"../models_evaluation/chance_of_loss/feature_importances_{target}.jpg", bbox_inches="tight")
plt.close()


# f.plot_stacked_feature_importances(
#     df_feature_importances_plot["logreg_importances"],
#     target_name=target,
#     model_names_plot = ("Logistic Regression"),
#     outfile=f"../models_evaluation/chance_of_loss/feature_importances_{target}.jpg"
# )


### Save final feature space 
## The final selection of features is used later for the non-parametric Bayesian Network

## drop records with missing target values
print(f"Dropping {df_candidates[f'{target}'].isna().sum()} records from entire dataset due that these values are nan in target variable")
df_candidates = df_candidates[ ~df_candidates[target].isna()]
print(f"Keeping {df_candidates.shape[0]} records and {df_candidates.shape[1]} features")


## sort features by their overall importance (weighted sum across across all features) 
final_feature_names = df_feature_importances_plot[f"{model_name}_importances"].sort_values(ascending=False).index##[:10]
print(final_feature_names)

## save importnat features, first column contains target variable
fs.save_selected_features(
    df_candidates, 
    pd.DataFrame(df_candidates, columns=[target]), 
    final_feature_names,
    filename=f"../selected_features/chance_of_loss/final_predictors_{target}.xlsx"
)


### Partial dependence
## PDP shows the marginal effect that one or two features have on the predicted outcome.

## store partial dependences for each model
pdp_features = {a : {} for a in [model_name]}

for model_name in [model_name]:

    Xy_pdp = eval_sets[model_name].dropna() #  solve bug on sklearn.partial_dependece() which can not deal with NAN values
    X_pdp, y_pdp = Xy_pdp[Xy_pdp.columns.drop(target)], Xy_pdp[target]
    X_pdp = pd.DataFrame(
        MinMaxScaler().fit_transform(X_pdp), # for same scaled pd plots across models
        columns=X.columns
        )
    Xy_pdp = pd.concat([y_pdp, X_pdp], axis=1)

    for predictor_name in X.columns.to_list(): 
        features_info =  {
            #"percentiles" : (0.05, .95) # causes NAN for some variables for XGB if (0, 1)
            "model" : final_models_trained[model_name], 
            "Xy" : Xy_pdp, 
            "y_name" : target, 
            "feature_name" : predictor_name, 
            "scale"  : True
        }         
        # get Partial dependences for sklearn models
        partial_dep = me.get_partial_dependence(**features_info)

        pdp_features[model_name][predictor_name] = partial_dep


## Plot PDP

most_important_features = df_feature_importances_plot.sort_values(f"{model_name}_importances", ascending=False).index

categorical = [] # ["flowvelocity", "further_variables .."]
ncols = 1
nrows = len(most_important_features[:10])
idx = 0

plt.figure(figsize=(5,25))
# plt.suptitle(f"Partial Dependences for {target}", fontsize=18, y=0.95)


## create PDP for all three models
for feature in most_important_features[:10]:
    for model_name, color, idx_col in zip([model_name], ["darkblue"], [0]):
        
        # idx position of subplot
        ax = plt.subplot(nrows, ncols, idx + 1 + idx_col)
        feature_info = {"color" : color, "ax" : ax} 

        # plot
        df_pd_feature = pdp_features[model_name][feature]  
        p = f.plot_partial_dependence(
            df_pd_feature, 
            feature_name=feature, 
            partial_dependence_name="yhat", 
            categorical=[],
            outfile=f"../models_evaluation/chance_of_loss/pdp_{target}.jpg",
            **feature_info
            )
        p

    idx = idx + 1


#plt.subplots_adjust(top=0.2)
plt.savefig(f"../models_evaluation/chance_of_loss/pdp_{target}.jpg", bbox_inches="tight")


# ### Empirical median ~ predicted median
# Compare median and mean of predicted  vs observed target values
for k,v in predicted_values.items():
    print(f"\n{k}")
    print(eu.empirical_vs_predicted(predicted_values[k]["y_true"], predicted_values[k]["y_pred"]))
