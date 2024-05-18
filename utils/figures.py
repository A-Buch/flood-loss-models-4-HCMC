#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure functions"""

__author__ = "Anna Buch, Heidelberg University"
__email__ = "a.buch@stud.uni-heidelberg.de"


import numpy as np
import pandas as pd
import contextlib

from sklearn.metrics import confusion_matrix, mean_absolute_error as mae
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# from matplotlib.cbook import boxplot_stats
from matplotlib.colors import to_rgba
import seaborn as sns

# import evaluation_utils as eu
import settings as s
import feature_selection as fs

from rpy2.robjects.packages import importr

caret = importr("caret")  # package version needs to be higher than  >=  6.0-90
party = importr("party")


logger = s.init_logger("__figures__")  # TODO impl in the rest of the functions


def plot_spearman_rank(df_corr, min_periods=100, signif=True, psig=0.05, target=None):
    """
    ## Code snippet modified: https://stackoverflow.com/questions/69900363/colour-statistically-non-significant-values-in-seaborn-heatmap-with-a-different

    df_corr (dataframe): dataframe with variables to plot
    min_periods (int): Minimum number of observations required per pair of columns to have a valid result
    signif (boolean): should non significant be masked
    psig (float): signifcance level
    return: Figure for Pearson Correlation
    """
    ## get the p value for spearman coefficient, subtract 1 on the diagonal
    pvals = df_corr.corr(method=lambda x, y: stats.spearmanr(x, y)[1], min_periods=min_periods) - np.eye(
        *df_corr.corr(method="spearman", min_periods=min_periods).shape
    )  # np.eye(): diagonal= ones, elsewere=zeros

    #  main plot
    sns.heatmap(
        df_corr.corr(method="spearman", min_periods=min_periods),
        annot=True,
        square=True,
        center=0,
        cmap="RdBu",
        fmt=".2f",
        zorder=1,
        annot_kws={"size": 10},
    )
    plt.title(f"Spearman's rank correlation: {target}", fontsize=16)

    # signifcance mask
    if signif:
        ## add another heatmap with colouring the non-significant cells
        sns.heatmap(
            df_corr.corr(method="spearman", min_periods=min_periods)[pvals >= psig],
            annot=False,
            square=True,
            cbar=False,
            ## add-ons
            cmap=sns.color_palette("Greys", n_colors=1, desat=1),
            zorder=2,
        )  # put the map above the heatmap
        ## add a label for the colour
        colors = [sns.color_palette("Greys", n_colors=1, desat=1)[0]]
        texts = [f"not significant (at {psig})"]
        patches = [mpatches.Patch(color=colors[i], label="{:s}".format(texts[i])) for i in range(len(texts))]
        plt.legend(handles=patches, bbox_to_anchor=(0.85, 1.05), loc="center")


def corrdot(*args, **kwargs):
    corr_r = args[0].corr(args[1], method="spearman", min_periods=100)
    # corr_r = args[0].corr(args[1], 'pearson')
    corr_text = round(corr_r, 2)
    ax = plt.gca()
    # font_size = abs(corr_r) * 80 + 5
    font_size = 26  # make fontsize readable
    ax.annotate(
        corr_text,
        [
            0.5,
            0.5,
        ],
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=font_size,
    )


def corrfunc(x, y, **kws):
    r, p = stats.spearmanr(x, y)
    p_stars = ""
    if p <= 0.05:
        p_stars = "*"
    if p <= 0.01:
        p_stars = "**"
    if p <= 0.001:
        p_stars = "***"
    ax = plt.gca()
    ax.annotate(p_stars, xy=(0.65, 0.6), xycoords=ax.transAxes, color="red", fontsize=70)


def plot_correlations(df, outfile=None, impute_na=False):
    """
    Correlations visualized by Scatterplots, significance and freuqnecy plots between all variables
    df : pd.DataFrame
    impute_na (bool): impute missing values to see better outliers, as it would be with removing the entire record
    """
    if impute_na:
        logger.info("imputing columns with median")
        df = df.apply(lambda x: x.fillna(x.median()), axis=0)
    else:
        df = df.dropna()
        logger.info(f"removing {len(df[df.isna()])} records with missing data")

    sns.set(style="white", font_scale=1.6)
    g = sns.PairGrid(df, aspect=1.5, diag_sharey=False, despine=False)
    g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={"color": "red", "lw": 1}, scatter_kws={"color": "black", "s": 20})
    g.map_diag(
        sns.distplot,
        color="black",
        kde_kws={"color": "red", "cut": 0.7, "lw": 1},
        # hist_kws={'histtype': 'bar', 'lw': 2, #'bins': 'auto', # 10
        #             'edgecolor': 'k', 'facecolor':'grey'}
    )
    g.map_diag(sns.rugplot, color="black")
    g.map_upper(corrdot)
    g.map_upper(corrfunc)
    g.fig.subplots_adjust(wspace=0, hspace=0)

    # Remove axis labels
    for ax in g.axes.flatten():
        ax.set_ylabel("")
        ax.set_xlabel("")

    # Add titles to the diagonal axes/subplots
    for ax, col in zip(np.diag(g.axes), df.columns):
        ax.set_title(col, y=0.82, fontsize=26)
    try:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
    except:
        pass


def plot_confusion_matrix(y_true, y_pred, outfile):
    """
    Plot confusion matrix
    y_true (pd.Series): observed target values
    y_pred (pd.Series or np.array): predicted target values
    outfile (str): Location to store plot
    return: saved figure if assigned and pd.DataFrame with confusion matrix
    """
    # cm = pd.DataFrame(confusion_matrix(y_true, y_pred))
    # cm.set_index("true_" + cm.index.astype(str), inplace=True)
    # cm = cm.add_prefix("pred_")

    # plt.figure(figsize=(6,6))
    # sns.set(font_scale=1.5)
    # sns.heatmap(
    #     cm,
    #     fmt="g", cmap="Blues",
    #     square=True,
    #     annot=True, annot_kws={"size":16},
    #     cbar=True,
    # )
    plt.figure(figsize=(6, 6))
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    sns.set(font_scale=1.5)
    sns.heatmap(
        cm,
        fmt=".2f",
        cmap="Blues",
        square=True,
        annot=True,
        annot_kws={"size": 16},
        cbar=True,
    )
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    plt.show(block=False)

    with contextlib.suppress(Exception):
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
    return cm


def plot_learning_curves(model, train_set, test_set, target, outfile, model_name=None):
    """
    Plot learning curve of sklearn model
    Code snippet: https://nvsyashwanth.github.io/machinelearningmaster/learning-curves/
    """

    fig = plt.figure(figsize=(5, 5))
    plt.style.use("seaborn-v0_8-white")

    X_train = train_set.drop(target, axis=1)
    y_train = train_set[target]
    X_test = test_set.drop(target, axis=1)
    y_test = test_set[target]

    train_errors = []
    test_errors = []

    for i in range(1, len(X_train)):
        model.fit(X_train[:i], y_train[:i])
        y_train_pred = model.predict(X_train[:i])
        y_test_pred = model.predict(X_test)
        train_errors.append(mae(y_train_pred, y_train[:i]))
        test_errors.append(mae(y_test_pred, y_test))

    plt.plot(range(1, len(X_train)), train_errors, label="Training error", color="blue")
    plt.plot(range(1, len(X_train)), test_errors, label="Test error", color="red")
    plt.ylim(0, 25)  # limit plot to < 25 MAE for uniform scales across the models

    plt.title(f"Learning curve for {model_name}")
    plt.xlabel("Number of samples in the training set")
    plt.ylabel("MAE")
    plt.legend()
    plt.close()

    fig.get_figure().savefig(outfile, dpi=300, bbox_inches="tight")


# TODO impl plot_r_learning_curve() inside plot_learning_curve()
def plot_r_learning_curve(eval_set, target_name, outfile, r_model_name="cforest"):
    """
    eval_set: pd.DataFrame with entire dataset (test and train sets) incl target
    """
    ## calc learning curve of cforest
    r_df_learning_curve = fs.r_dataframe_to_pandas(
        caret.learning_curve_dat(
            eval_set,
            target_name,
            proportion=np.arange(0.1, 1, 0.05),  # proportion  of samples used for train set
            method="cforest",
            metric="MAE",
            trControl=caret.trainControl(method="repeatedcv", number=5, repeats=1),
            verbose=False,
        )
    )

    ##  take mean of cross validated training sample sizes
    test_set = r_df_learning_curve.loc[r_df_learning_curve["Data"] == "Resampling", :]
    test_errors = test_set.groupby(["Training_Size"])["MAE"].mean().to_list()

    ## train and test errors and sizes for plotting
    train_set = r_df_learning_curve.loc[r_df_learning_curve["Data"] == "Training", :]
    train_errors = train_set.loc[train_set["Data"] == "Training", "MAE"].to_list()
    train_sizes = train_set.loc[train_set["Data"] == "Training", "Training_Size"]

    ## plot learning curve
    fig = plt.figure(figsize=(5, 5))
    plt.style.use("seaborn-v0_8-white")

    plt.plot(range(0, len(train_sizes)), train_errors[:], label="Training error", color="blue")
    plt.plot(range(0, len(train_set)), test_errors[:], label="Test error", color="red")
    plt.title(f"Learning curve for {r_model_name}")
    plt.xlabel("Number of samples in the training set")
    plt.xticks(range(0, len(train_sizes)), train_set["Training_Size"].to_list())
    plt.ylabel("MAE")
    plt.legend()
    plt.close()

    fig.get_figure().savefig(outfile, dpi=300, bbox_inches="tight")


def plot_stacked_feature_importances(df_feature_importances, target_name, model_names_plot, outfile):
    """
    Stack feature importances of multiple models into one barchart
    df_feature_importances : pd.DataFrame with columns which contain feature importances to plot
    """
    model_name_1, model_name_2, model_name_3 = model_names_plot  # TODO update with s.color_palette_models from settings

    ## TODO remove hardcode - function is currently limited to three models
    feature_importance_1, feature_importance_2, feature_importance_3 = df_feature_importances.columns.to_list()

    ## plot
    plt.figure(figsize=(30, 22))  ## TODO add figsize as kwargs fig_kwargs={'figsize':[15,15]}
    sns.set_style("whitegrid", {"axes.grid": False})

    fig = df_feature_importances.plot.barh(
        stacked=True,
        color={feature_importance_1: "steelblue", feature_importance_2: "darkblue", feature_importance_3: "grey"},
        # color=s.color_palette_models,
        width=0.5,
        alpha=0.7,
    )
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.title(f"Feature importances for {target_name.replace('_',' ')}", fontweight="bold", fontsize=16)

    ## legend
    top_bar = mpatches.Patch(color="steelblue", label=model_name_1, alpha=0.7)  # TODO update with s.color_palette_models from settings
    middle_bar = mpatches.Patch(color="darkblue", label=model_name_2, alpha=0.7)
    bottom_bar = mpatches.Patch(color="grey", label=model_name_3, alpha=0.7)
    plt.tick_params(axis="x", which="major", labelsize=12)
    plt.tick_params(axis="y", which="major", labelsize=12)
    plt.legend(handles=[top_bar, middle_bar, bottom_bar], loc="lower right")
    plt.tight_layout()

    fig.get_figure().savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


def plot_partial_dependence(df_pd_feature, feature_name: str, partial_dependence_name: str, categorical: list, outfile, **kwargs):
    """
    Creates plots for partial dependencies for multiple models
    :param df_pd_feature: df_pd_feature
    :param feature_name: name of feature (x-axis)
    :param partial_dependence_name: name of target (y-axis)
    :param feature_names: List of features
    :param categorical (list): list of features which are categorical TODO make as boolean
    :return: plot and png of PDP
    """
    if feature_name in categorical:
        sns.barplot(data=df_pd_feature, x=df_pd_feature[feature_name], y=df_pd_feature[partial_dependence_name], **kwargs)
    else:
        sns.lineplot(data=df_pd_feature, x=df_pd_feature[feature_name], y=df_pd_feature[partial_dependence_name], legend=False, **kwargs)

    # kwargs["ax"].get_xaxis().set_visible(False)
    kwargs["ax"].set_xlabel(feature_name)
    kwargs["ax"].set_ylabel("")
    # ax.get_yaxis().set_visible(True)

    kwargs["ax"].tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom="on",  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom="off",  # labels along the bottom edge are off)
    )
    kwargs["ax"].tick_params(
        axis="y",
        which="both",
        left="on",
        right=False,
    )
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")


def plot_observed_predicted(
    y_true, y_pred, hue=None, hue_colors=("darkgrey", "steelblue"), xlabel="observed", ylabel="predicted", alpha=0.6, legend=False, outfile="test.png"
):
    """
    Scatter plot of observations vs predictions with optional class colors
    NOTE: hue is currently limited to binary cases
    # Code Snippet: https://stackoverflow.com/questions/66667334/python-seaborn-alpha-by-hue
    """

    sns.set(style="white", font_scale=1.2)

    color_dict = {
        0: to_rgba(hue_colors[0], alpha),  # set transparency for each class independently
        1: to_rgba(hue_colors[1], alpha),
    }
    if hue is None:
        color_dict = color_dict[1]

    g = sns.JointGrid(
        x=y_true,
        y=y_pred,
        hue=hue,
        height=5,
        space=0,
    )
    # g.plot_joint(sns.scatterplot, palette=color_dict, edgecolors=color_dict, legend=legend)
    p = sns.scatterplot(x=y_true, y=y_pred, hue=hue, palette=color_dict, edgecolors=color_dict, legend=legend, ax=g.ax_joint)

    if legend is True:
        # p.legend(fontsize=10, )  # outside plot: bbox_to_anchor= (1.2,1)
        plt.setp(p.get_legend().get_texts(), fontsize="12")
        plt.setp(p.get_legend().get_title(), fontsize="15")

    g.plot_marginals(sns.histplot, kde=True, palette=color_dict, fill=True)  # multiple='stack')
    # sns.displot(penguins, x="flipper_length_mm", kind="kde")

    g1 = sns.regplot(x=y_true, y=y_pred, line_kws={"lw": 1.0}, scatter=False, ax=g.ax_joint)
    regline = g1.get_lines()[0]
    regline.set_color("steelblue")

    x0, x1 = (0, 100)
    y0, y1 = (0, 100)
    lims = [min(x0, y0), max(x1, y1)]
    g.ax_joint.plot(
        lims,
        lims,
        c="black",
        lw=0.5,
    )  # equal line
    g.set_axis_labels(xlabel=xlabel, ylabel=ylabel)

    # plt.title(f"Observed and predicted {target}")
    # save plot
    plt.savefig(outfile, dpi=300, bbox_inches="tight")

    plt.show()


# plt.close()


def plot_residuals(df_residuals, model_names_abbreviation, model_names_plot, outfile):
    """
    Generate plots of residuals , TOdO write residuals to csv file
    residuals : model residuals, property from ModelEvaluation.residuals
    model_names_abbreviation (str): if None than plot directly is called, otherwise figures plotted next to each other grouped by model names
    model_names_plot  : TOdO
    NOT IMPL: feature_name (str ): name of feature to group residuals
    outfile (str): Path and file to store figures
    """

    sns.set_style(
        "white",
        rc={
            "xtick.bottom": True,
            "ytick.left": True,
        },
    )

    models_n = len(model_names_plot)

    ## TODO add figsize as kwargs fig_kwargs={'figsize':[15,15]}

    f, (ax0, ax1) = plt.subplots(2, models_n, figsize=(12, 8), sharex="col", sharey="row")

    for idx, abbrev, model_name, color in zip(range(0, models_n), model_names_abbreviation, model_names_plot, ["steelblue", "darkblue", "grey"]):
        y_true = df_residuals[abbrev]["y_true"]
        y_pred = df_residuals[abbrev]["y_pred"]
        residuals = df_residuals[abbrev]["residual"]

        ## 1.st plot obs ~ pred
        sns.scatterplot(x=y_true, y=y_pred, ax=ax0[idx], alpha=0.5, color=color)
        sns.regplot(x=y_true, y=y_pred, ax=ax0[idx], scatter=False)

        ax0[idx].set_title(f"{model_name} regression")
        ax0[idx].set_ylabel("predictions")

        ## 2.nd obs ~ residuals
        sns.scatterplot(x=y_true, y=residuals, ax=ax1[idx], alpha=0.5, color=color)
        # ax1[idx].set_title(f"{full_name} regression ")
        ax1[idx].set_xlabel("observations")
        ax1[idx].set_ylabel("residuals [prediction - observation]")
        ax1[idx].axhline(0, ls="--")

        plt.suptitle("Residual distributions of the best-performed estimators", fontweight="bold", fontsize=25)
        plt.subplots_adjust(top=0.2)

        f.get_figure().savefig(outfile, dpi=300, bbox_inches="tight")
        plt.tight_layout()
        plt.close()


def boxplot_outer_scores_ncv(models_scores, outfile, target_name):
    # TODO make plot with flexible number of models, currently limited to 3 models or one LogisticReg
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
    names = df_outer_scores_of_all_models.columns.drop("modelname")
    ncols = len(names)
    fig, axes = plt.subplots(1, ncols, figsize=(25, 10), layout="constrained")

    ## TODO add figsize as kwargs fig_kwargs={'figsize':[15,15]}

    # plot
    for name, ax in zip(names, axes.flatten()):
        try:
            ax.set_title(name.split("_")[1], fontsize=25)
        except:
            ax.set_title(name, fontsize=25)

        ## for all metrices except R2
        if name != "test_R2":
            sns.set_style("whitegrid", {"grid.linestyle": ":"})

            try:  # TODO make plot with flexible number of models
                sns.boxplot(
                    y=name,
                    x="modelname",
                    data=df_outer_scores_of_all_models,
                    orient="v",
                    ax=ax,
                    palette=s.color_palette_models,
                    width=[0.4],
                    boxprops=dict(alpha=0.7),
                ).set(xlabel=None, ylabel=None)

            except KeyError or TypeError:  # if only one model
                sns.boxplot(
                    y=name,
                    x="modelname",
                    data=df_outer_scores_of_all_models,
                    orient="v",
                    ax=ax,
                    palette={df_outer_scores_of_all_models["modelname"][0]: "darkblue"},
                    width=[0.4],
                    boxprops=dict(alpha=0.7),
                ).set(xlabel=None, ylabel=None)

        ax.set_xticks([])  # surpress x tick labels of model names
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=25)

        # regression task
        if ("modelname" != "LogisticRegression") or ("modelname" != "RandomForestClassifier"):
            (ax.get_shared_y_axes().join(axes[0], *axes[:-1]),)  # share y axis except for SMAPE
            ax.axhline(y=0, color="black", linewidth=0.8, alpha=0.7, ls="--")  # add zero-line
        # classificaion task
        else:
            (ax.get_shared_y_axes().join(axes[0], *axes[:]),)  # share all y axis (only for logReg)
            ax.axhline(y=0.5, color="black", linewidth=0.8, alpha=0.7, ls="--")  # add 50%-line

        ax.tick_params(axis="y", labelsize=20)
        ax.set_title(name.split("_")[1], fontsize=20)
        # ax.xlabel("")

        plt.suptitle(
            f"Prediction errors of the best-performed estimators for {target_name}, assessed by nested cross-validation",
            fontweight="bold",
            fontsize=25,
        )
        plt.subplots_adjust(top=0.955)

        ## legend
        top_bar = mpatches.Patch(color="steelblue", label="Elastic Net", alpha=0.7)  # TODO update with s.color_palette_models from settings
        middle_bar = mpatches.Patch(color="darkblue", label="Conditional Random Forest", alpha=0.7)
        bottom_bar = mpatches.Patch(color="grey", label="XGBRegressor", alpha=0.7)
        plt.legend(handles=[top_bar, middle_bar, bottom_bar], fontsize=20, loc="lower center", bbox_to_anchor=(0.5, 0.1))
        # ax.title(name, fontsize=14)

        plt.tight_layout()
        plt.savefig(outfile, dpi=300, bbox_inches="tight")  # format='jpg'


def plot_boxplot_scatterplot(df, group, column, scatterpoints):
    # TODO generalize function , eg. xlabel
    """ """
    all_input_p = df
    grouped = all_input_p.groupby(group)
    categories = np.unique(all_input_p[scatterpoints])
    colors = np.linspace(0, 1, len(categories))
    colordict = dict(zip(categories, colors))
    all_input_p[scatterpoints] = all_input_p[scatterpoints].apply(lambda x: colordict[x])

    names, vals, xs, colrs = [], [], [], []

    for i, (name, subdf) in enumerate(grouped):
        names.append(name)
        vals.append(subdf[column].tolist())
        xs.append(np.random.normal(i + 1, 0.04, subdf.shape[0]))
        colrs.append(subdf[scatterpoints].tolist())

    fig = plt.figure()
    ax = fig.add_subplot(111)

    p = ax.boxplot(
        vals,
        labels=names,
        # width=.5,
        showfliers=False,
        patch_artist=True,
        # boxprops={"facecolor":(1,0,0,.2), "edgecolor":'k'}
        boxprops={"facecolor": "steelblue", "alpha": 0.4},
        whiskerprops={"color": "k", "alpha": 0.4},
        capprops={"color": "k", "alpha": 0.4},
    )
    for patch, color in zip(p["boxes"], colors):
        # patch.set_facecolor("steelblue",)
        patch.set_edgecolor("black")

    # ngroup = len(vals)
    # clevels = np.linspace(0., 1., ngroup)

    for x, val, colr in zip(xs, vals, colrs):
        print(len(colr))
        # plt.scatter(x, val, c=cm.prism(clevel), alpha=0.4)
        # plt.scatter(x, val, c=colr, alpha=1., marker="x")
        sns.scatterplot(
            x=x,
            y=val,
            hue=colr,
            edgecolors=colr,
            marker="o",
            s=20,
            lw=0,
            # marker="x", s=60, lw=3,
            legend=False,
            palette=["green", "blue", "red"],
            # legend=True, palette=["green", "blue", "red"]
        )
    plt.xlabel("monhtly sale rates [€]")
    plt.ylabel(f"{column}")
    # plt.ylabel("fraction of implemented non-structural measures")
    # plt.ylabel("fraction of implemented emergency measures")

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
    # plt.close(fig)

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
