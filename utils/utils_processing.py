#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions for data cleaning"""


import numpy as np


def drop_object_columns(df):
    """
    Remove object columns from dataframe
    """
    df = df.loc[:, ~df.columns.str.contains(
            r"(.88)$|(.99)$|(.specify)$|(.Specify)$|(others)"
            )
        ] 
    return df


def drop_typos(df):
    """
    Repair typos in numeric columns
    """
    df = df.replace({
        " ": np.nan, 
        "":np.nan
        }) # fill empty cells, otherwise no cols append possible
    df = df.replace({"^,":"0.", ",":"."}, regex=True) 
    return df


def check_types(x):
    """
    Converts strings to floats, the ones that cannot be converted are returned as None
    :param x: Variable to be converted
    :return:
    """

    if isinstance(x, str):
        if x.isnumeric():
            return float(x)
    else:
        return x
