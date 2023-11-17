#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure functions"""

__author__ = "Anna Buch, Heidelberg University"
__email__ = "a.buch@stud.uni-heidelberg.de"


import numpy as np
import pandas as pd
import contextlib

from sklearn.metrics import confusion_matrix, PredictionErrorDisplay
from scipy import stats


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cbook import boxplot_stats  
import seaborn as sns

import utils.evaluation_metrics as em
import utils.settings as s

logger = s.init_logger("__figures__")



def plot_spearman_rank(df_corr, min_periods=100, signif=True, psig=0.05):
        """
        ## Code snippet modified: https://stackoverflow.com/questions/69900363/colour-statistically-non-significant-values-in-seaborn-heatmap-with-a-different

        df_corr (dataframe): dataframe with variables to plot
        min_periods (int): Minimum number of observations required per pair of columns to have a valid result
        signif (boolean): should non significant be masked
        psig (float): signifcance level
        return: Figure for Pearson Correlation 
        """ 
 
        ## get the p value for pearson coefficient, subtract 1 on the diagonal
        pvals = df_corr.corr(
            method=lambda x, y: stats.spearmanr(x, y)[1], min_periods=min_periods
            ) - np.eye(
                *df_corr.corr(method="spearman", min_periods=min_periods).shape
                )  # np.eye(): diagonal= ones, elsewere=zeros

        #  main plot
        sns.heatmap(
            df_corr.corr(method="spearman", min_periods=min_periods), 
            annot=False, square=True, 
            center=0, cmap="RdBu", 
            fmt=".2f", zorder=1,
        )

        # signifcance mask
        if signif:
                ## add another heatmap with colouring the non-significant cells
                sns.heatmap(df_corr.corr(method="spearman", min_periods=min_periods)[pvals>=psig], 
                            annot=False, square=True, cbar=False,
                            ## add-ons
                            cmap=sns.color_palette("Greys", n_colors=1, desat=1),  
                            zorder = 2) # put the map above the heatmap
        ## add a label for the colour
        colors = [sns.color_palette("Greys", n_colors=1, desat=1)[0]]
        texts = [f"not significant (at {psig})"]
        patches = [ mpatches.Patch(color=colors[i], label="{:s}".format(texts[i]) ) for i in range(len(texts)) ]
        plt.legend(handles=patches, bbox_to_anchor=(.85, 1.05), loc='center')


def corrdot(*args, **kwargs):
    corr_r = args[0].corr(args[1], method="spearman", min_periods=100)
    #corr_r = args[0].corr(args[1], 'pearson')
    corr_text = round(corr_r, 2)
    ax = plt.gca()
    font_size = abs(corr_r) * 80 + 5
    ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                ha='center', va='center', fontsize=font_size)

def corrfunc(x, y, **kws):
    r, p = stats.spearmanr(x, y)
    p_stars = ''
    if p <= 0.05:
        p_stars = '*'
    if p <= 0.01:
        p_stars = '**'
    if p <= 0.001:
        p_stars = '***'
    ax = plt.gca()
    ax.annotate(p_stars, xy=(0.65, 0.6), xycoords=ax.transAxes,
                color='red', fontsize=70)


def plot_correlations(df, impute_na=True):
    """
    Correlations visualized by Scatterplots, significance and freuqnecy plots between all variables
    df : pd.DataFrame
    impute_na (bool): impute missing values to see better outliers, as it would be with removing the entire record 
    """
    if impute_na:
        logger.info("imputing columns with median")
        df = df.apply(lambda x: x.fillna(x.median()),axis=0)
    else:
        df = df.dropna()
        logger.info(f"removing {len(df[df.isna()])} records with missing data")


    sns.set(style='white', font_scale=1.6)

    g = sns.PairGrid(df, aspect=1.5, diag_sharey=False, despine=False)
    g.map_lower(sns.regplot, lowess=True, ci=False,
                line_kws={'color': 'red', 'lw': 1},
                scatter_kws={'color': 'black', 's': 20})
    g.map_diag(sns.distplot, color='black',
            kde_kws={'color': 'red', 'cut': 0.7, 'lw': 1},
            hist_kws={'histtype': 'bar', 'lw': 2, #'bins': 'auto', # 10
                        'edgecolor': 'k', 'facecolor':'grey'})
    g.map_diag(sns.rugplot, color='black')
    g.map_upper(corrdot)
    g.map_upper(corrfunc)
    g.fig.subplots_adjust(wspace=0, hspace=0)

    # Remove axis labels
    for ax in g.axes.flatten():
        ax.set_ylabel('')
        ax.set_xlabel('')

    # Add titles to the diagonal axes/subplots
    for ax, col in zip(np.diag(g.axes), df.columns):
        ax.set_title(col, y=0.82, fontsize=26)



def plot_confusion_matrix(y_true, y_pred, outfile):
    """
    Plot confusion matrix
    y_true (pd.Series): observed target values
    y_pred (pd.Series or np.array): predicted target values
    outfile (str): Location to store plot
    return: saved figure if assigned and pd.DataFrame with confusion matrix
    """
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred))
    cm.set_index("true_" + cm.index.astype(str), inplace=True)
    cm = cm.add_prefix("pred_")

    plt.figure(figsize=(6,6))
    sns.set(font_scale=1.5)
    sns.heatmap(
        cm,
        fmt="g", cmap="Blues", 
        square=True, 
        annot=True, annot_kws={"size":16},
        cbar=True,
    )
    with contextlib.suppress(Exception):
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
    return cm



def plot_stacked_feature_importances(df_feature_importances, target_name, model_names_plot, outfile):
    """
    Stack feature importances of multiple models into one barchart
    df_feature_importances : pd.DatFrame with columns which contain feature importances to plot
    """
    model_name_1, model_name_2 , model_name_3 = model_names_plot # TODO update with s.color_palette_models from settings

    ## TODO remove hardcode - function is currently limited to three models
    feature_importance_1, feature_importance_2, feature_importance_3 = df_feature_importances.columns.to_list()

    ## plot
    plt.figure(figsize=(30, 22))
    fig = df_feature_importances.plot.barh(
        stacked=True, 
        color={feature_importance_1:"darkblue", feature_importance_2:"steelblue", feature_importance_3:"grey"},
        # color=s.color_palette_models,
        width=.5, alpha=.7,
    )
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.title(f"Feature Importances for {target_name.replace('_',' ')}")

    ## legend
    top_bar = mpatches.Patch(color="darkblue", label=model_name_1)  #TODO update with s.color_palette_models from settings
    middle_bar = mpatches.Patch(color="steelblue", label=model_name_2)
    bottom_bar = mpatches.Patch(color="grey", label=model_name_3)
    plt.tick_params(axis='x', which='major', labelsize=12)
    plt.tick_params(axis='y', which='major', labelsize=12)
    plt.legend(handles=[top_bar, middle_bar, bottom_bar], loc="lower right")
    plt.tight_layout()
    
    fig.get_figure().savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

    

def plot_partial_dependence(df_pd_feature, feature_name:str, partial_dependence_name:str, categorical:list, outfile, **kwargs):
    """
    Creates plots for partial dependecies for multiple models
    :param model: Model instance
    :param X_train: 
    :param feature_names: List of features
    :param categorical (list): list of features which are categorical
    :return:
    """
    if feature_name in categorical:
        sns.barplot(
            data=df_pd_feature, 
            x=df_pd_feature[feature_name], 
            y=df_pd_feature[partial_dependence_name], 
            **kwargs
        )
        sns.rugplot(df_pd_feature, x=df_pd_feature[feature_name], 
                    y=df_pd_feature[partial_dependence_name], 
                    height=-.02)
        # sns.rugplot(df_pd_feature, x=feature_name, y="yhat", height=-.02)
    else:      
        sns.lineplot(
            data=df_pd_feature, 
            x=df_pd_feature[feature_name], 
            y=df_pd_feature[partial_dependence_name], 
            legend=False,
            **kwargs
        )
        sns.rugplot(df_pd_feature, x=df_pd_feature[feature_name], 
                    y=df_pd_feature[partial_dependence_name], 
                    height=-.02)

    #kwargs["ax"].get_xaxis().set_visible(False)
    kwargs["ax"].set_xlabel("")
    kwargs["ax"].set_ylabel(feature_name)
    # ax.get_yaxis().set_visible(True)
    kwargs["ax"].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='on',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off'  # labels along the bottom edge are off)
        )
    kwargs["ax"].tick_params(
        axis='y',
        which='both',
        left='on',
        right='on',
    )
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    
   
def plot_residuals(residuals, model_names_abbreviation,  model_names_plot, outfile):
    """
    Generate plots of residuals and write residuals to csv file
    residuals : model residuals, property from ModelEvaluation.residuals
    X : pd.DataFrame with predictors
    feature_name (str ): name of feature to group residuals
    model_name (str): model's name
    out_dir (str): Path to store figures and csv file
    """
    models_n = len(model_names_plot)
                        
    f, (ax0, ax1) = plt.subplots( 2, models_n, figsize=(18, 8)) # sharey="row", 

    for idx, abbrev, full_name in zip(range(0, models_n), model_names_abbreviation, model_names_plot):
        
        y_true = residuals[abbrev]["y_true"] 
        y_pred = residuals[abbrev]["y_pred"]

        PredictionErrorDisplay.from_predictions(
            y_true,
            y_pred,
            kind="actual_vs_predicted",
            ax=ax0[idx],
            scatter_kwargs={"alpha":.5},
        )
        ax0[idx].set_title(f"{full_name} regression ")


        # Add the score in the legend of each axis
        for name, score in em.compute_score(y_true, y_pred).items():
            ax0[idx].plot([], [], " ", label=f"{name}={score}")
            ax0[idx].legend(loc="upper right")

        # plot the residuals vs the predicted values
        PredictionErrorDisplay.from_predictions(
            y_true,
            y_pred,
            kind="residual_vs_predicted",
            ax=ax1[idx],
            scatter_kwargs={"alpha":.5},
        )
        ax1[idx].set_title(f"{full_name} regression ")

        f.get_figure().savefig(outfile, dpi=300, bbox_inches="tight")

        plt.subplots_adjust(top=0.2)
        plt.tight_layout()
        plt.close()


def boxplot_outer_scores_ncv(models_scores, outfile):
    """
    Boxplot grouped by evalatuation metrics (eg MAE, RMSE..)
    models_scores (dict): nested dictionary with outer model scores
    outfile (str): outfile and path
    """
    ## collect performance scores from outer folds of nested cross validation
    df_outer_scores_of_all_models = pd.DataFrame()
    for model_name in list(models_scores.keys()):
        outer_scores = pd.DataFrame(models_scores[model_name].copy())
        outer_scores["modelname"] = model_name
        df_outer_scores_of_all_models = pd.concat([df_outer_scores_of_all_models, outer_scores], axis=0)

    ## settings for plot
    names = df_outer_scores_of_all_models.columns.drop('modelname')
    ncols = len(names)
    fig, axes = plt.subplots(1, ncols, figsize=(25, 10), layout="constrained")

    # plot
    for name, ax in zip(names, axes.flatten()):
        try:
            ax.set_title(name.split("_")[1], fontsize=25)
        except:
            ax.set_title(name, fontsize=25)
        
        ## for all metrices except R2
        if name != "test_R2":
            sns.set_style("whitegrid")
            sns.boxplot(
                y=name, x="modelname", 
                data=df_outer_scores_of_all_models, 
                orient='v', ax=ax, 
                palette=s.color_palette_models,
                width=[0.4], boxprops=dict(alpha=.7),
            ).set(xlabel=None, ylabel=None)
        
        ## only for R2 due that it has ofte outliers which leads to a large value-range on y-axis with small boxplots
        else: 
            sns.set_style("whitegrid")
            sns.boxplot(
                y=name, x="modelname", 
                data=df_outer_scores_of_all_models, 
                orient='v', ax=ax, 
                palette=s.color_palette_models, 
                width=[0.4], showfliers=False, boxprops=dict(alpha=.7),
            ).set(xlabel=None, ylabel=None)

            ## report not visualized Rs scores aka outliers
            boxplot_stats(df_outer_scores_of_all_models["test_R2"]).pop(0)['fliers']  ## outleirs from R2
            logger.info(f"removed {len(np.array([0.4501175, 4501175]))} outliers to improve visualization of R2 value-range")

        ax.set_xlabel(None, fontsize=35)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=25)
        ax.get_shared_y_axes().join(axes[0], *axes[:-1]),  # share y axis except for SMAPE
        ax.axhline(y=0, color='black', linewidth=.8, alpha=.5, ls='--')  # add zero-line

        # fig.tight_layout()
        plt.tight_layout()
        plt.savefig(outfile, dpi=300, bbox_inches="tight")  #format='jpg'
        


    # ## Plot scatter plot of residuals by variable
    # ## logger.info("Generating scatter plot of residuals ...")
    # cols = 3
    # rows = math.ceil(len(residuals[feature_name].unique()) / cols)
    # fig, axes = plt.subplots(
    #     rows, cols, figsize=(10, 15), constrained_layout=True, sharex=True, sharey=True
    # )
    # ## Join variables to group to residuals
    # residuals = residuals.join(X.loc[:, feature_name], how="left")
    # residuals.round(2).to_csv(
    # Path(out_dir) / f"residuals_{model_names_abbreviation}.csv", index=False
    # )   
    # ## scatterplot of residuals grouped by Variable
    # for (var, group), ax in zip(residuals.groupby(feature_name), axes.flatten()):
    #     group.plot(x="y_true", y="y_pred", kind="scatter", ax=ax, title=var)
    #     ax.plot([0, 10], [0, 10], color="grey", linestyle="--", alpha=0.4)
    #     ax.set_xlabel("Observed")
    #     ax.set_ylabel("Predicted")

    # fig.suptitle(f"Predicted {target} vs. Observed {target}")
    # plt.savefig(
    #     Path(out_dir) / f"scatter_{target}_{model_name}.jpg", bbox_inches="tight"
    # )
    #plt.close(fig)

    # ## boxplot of residuals grouped by other predictor
    # # logger.info("Generating box plot of residuals...")
    # fig, ax = plt.subplots(figsize=(15, 8), constrained_layout=True)
    # sns.boxenplot(y=feature_name, x="residuals", data=residuals, orient="h", ax=ax)
    # plt.axvline(x=-10, color="grey", linestyle="--")
    # plt.axvline(x=10, color="grey", linestyle="--")
    # plt.axvline(x=0, color="grey", linestyle="--")
    # fig.suptitle(f"Residuals of predicted {target} vs. observed {target} by {feature_name} categories")
    # plt.xlabel("Residuals [%]")
    # plt.savefig(
    #     Path(out_dir) / f"residuals_{feature_name}_{model_names_plot}.jpg", bbox_inches="tight"
    # )
    # plt.close(fig)
