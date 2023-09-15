#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions for preprocessing"""

import os
import numpy as np
import json


def load_config(config_file):
    """
    Load config file
    :param config_file: path to config file (str)
    :return:
    """
    assert os.path.exists(
        config_file
    ), f"Configuration file does not exist: {os.path.abspath(config_file)}"
    with open(config_file, "r") as src:
        config = json.load(src)
    return config

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
