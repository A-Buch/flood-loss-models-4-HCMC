#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions for preprocessing"""

import os
import numpy as np
import pandas as pd
import json

from dataclasses import dataclass
import difflib


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


@dataclass()
class FuzzyMerge:
    """
        Works like pandas merge except also merges on approximate matches.
        modified from: https://stackoverflow.com/questions/74778263/python-merge-two-dataframe-based-on-text-similarity-of-their-columns
    """
    left: pd.DataFrame
    right: pd.DataFrame
    left_on: str
    right_on: str
    how: str = "left" # "inner"  
    n: int = 1  # match with best one
    cutoff: float = 0.6  #  higher cutoff == more strict in matching, TODO make cutoff as variable,

    def main(self) -> pd.DataFrame:
        temp = self.right.copy()
        temp[self.left_on] = [
            self.get_closest_match(x, self.left[self.left_on]) for x in temp[self.right_on]
        ]

        return self.left.merge(temp, on=self.left_on, how=self.how)

    def get_closest_match(self, left: pd.Series, right: pd.Series) -> str or None:
        matches = difflib.get_close_matches(left, right, n=self.n, cutoff=self.cutoff)

        return matches[0] if matches else None
        
