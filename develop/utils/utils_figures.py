#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure functions"""


import numpy as np

from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import pearsonr

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns


def plot_scatterplot_3d(df, x, y, z):
    """
    df (dataframe): dataframe with variables to plot
    x (str): column to plot on x axis
    y (str): column to plot on y axis
    z (str): column to plot as colors
    return: Figure for Scatterplot with two independent variables 
    """         
    sns.relplot(data=df, x=df[x], y=df[y], hue=df[z]) #aspect=1.61)


def plot_pearsoncorrelation(df_sm_corr, signif=True, psig=0.05):
        """
        ## Code snippet : https://stackoverflow.com/questions/69900363/colour-statistically-non-significant-values-in-seaborn-heatmap-with-a-different

        df_sm_corr (dataframe): dataframe with variables to plot
        signifcance (boolean): should non significant be masked
        psig (float): signifcance level
        return: Figure for Pearson Correlation 
        """ 
        
        ## get the p value for pearson coefficient, subtract 1 on the diagonal
        pvals = df_sm_corr.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*df_sm_corr.corr().shape)
        psig = 0.05  # 5% signif level

        #  main plot
        sns.heatmap(df_sm_corr.corr(), annot=False, square=True, cmap="RdBu", fmt=".2f", zorder=1)

        # signifcance mask
        if signif == True:
                ## add another heatmap with colouring the non-significant cells
                sns.heatmap(df_sm_corr.corr()[pvals>=psig], annot=False, square=True, cbar=False, 

        ## add-ons
        cmap=sns.color_palette("Greys", n_colors=1, desat=1),  zorder = 2) #put the map above the heatmap
        ## add a label for the colour
        colors = [sns.color_palette("Greys", n_colors=1, desat=1)[0]]
        texts = [f"not significant (at {psig})"]
        patches = [ mpatches.Patch(color=colors[i], label="{:s}".format(texts[i]) ) for i in range(len(texts)) ]
        plt.legend(handles=patches, bbox_to_anchor=(.85, 1.05), loc='center')



def plot_feature_importance(coefs, n=10, figure_size=(20, 15), target="_ "):
    """
    coefs: pd.Series or Dataframe with feautre names and coefficients 
    n (int): show only the n-th most important features 
    figure_size (tuple): height and width of plot
    target =
    Return: feature importance plot
    """
    plt.rcdefaults()
    fig, ax = plt.subplots()
    fig.size = figure_size

    y_pos = np.arange(len(coefs.index))

    ax.barh(y_pos[0:10], coefs.values[0:n])
    ax.set_yticks(y_pos[0:n])
    ax.set_yticklabels(coefs.index[0:n])
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Coefficents', size=14)
    ax.set_ylabel('Features', size=14)
    ax.set_title(f'Conditinal Inference Tree \nFeature Importance for {target.split("_")[1]}', size=20)
    #plt.grid(axis='y')
    #plt.grid(axis='x')
    ax.set_axisbelow(True)
    for index, value in enumerate(np.around(coefs.values[0:n],1)):
        plt.text(value, index, str(value))
    plt.show()

    