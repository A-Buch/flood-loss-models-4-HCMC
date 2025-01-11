#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions for preprocessing"""

__author__ = "Anna Buch, Heidelberg University"
__email__ = "anna.buch@uni-heidelberg.de"


import os
import numpy as np
import pandas as pd
import json

from dataclasses import dataclass
import difflib
import contextlib

import src.settings as s

logger = s.init_logger("__preprocessing__")


def create_output_dir(output_dir):
    """
    Create path to store outputs as pathlib objects
    return: string of created new output path (relative to cwd)
    """
    with contextlib.suppress(Exception):  # preserve directory and its file if it already exists
        print("Create ", output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
    return os.path.relpath(output_dir)


def load_config(config_file: str):
    """
    Load e.g. hyperparameter files
    """
    assert os.path.exists(config_file), f"Could not find file: {os.path.abspath(config_file)}"
    with open(config_file, "r") as src:
        config = json.load(src)
    return config


def drop_object_columns(df):
    """
    Remove object columns from dataframe
    """
    df = df.loc[:, ~df.columns.str.contains(r"(.88)$|(.99)$|(.specify)$|(.Specify)$|(others)")]
    return df


def drop_typos(df):
    """
    Repair typos in numeric columns
    """
    df = df.replace({" ": np.nan, "": np.nan})  # fill empty cells, otherwise no cols append possible
    df = df.replace({"^,": "0.", ",": "."}, regex=True)
    return df


# def check_types(x):
#     """
#     Converts strings to floats, the ones that cannot be converted are returned as None
#     :param x: Variable to be converted
#     :return:
#     """
#     if not isinstance(x, str):
#         return x
#     if x.isnumeric():
#         return float(x)


def percentage_of_nan(df):
    """
    Print number of missing data per variable
    df : pd.DataFrame to derive amount of missing data per variable
    """
    return logger.info(f"Percentage of missing values per feature [%]\n {round(df.isna().mean().sort_values(ascending=False)[:15]  * 100)}")


@dataclass(frozen=False)  # frozen=False : make annoutations such as "cutoff" mutable
class FuzzyMerge:
    """
    Works like pandas merge except also merges on approximate matches.
    Dataclass is a class mainly to store data, unlike than a normal Class
    modified bassed on: https://stackoverflow.com/questions/74778263/python-merge-two-dataframe-based-on-text-similarity-of-their-columns
    """

    left: pd.DataFrame
    right: pd.DataFrame
    left_on: str
    right_on: str
    how: str = "left"  # "inner"
    n: int = 1  # match with best one
    cutoff: float = 0.9
    # higher cutoff == more strict in matching, TODO make cutoff as variable,

    def main(self) -> pd.DataFrame:
        df = self.right.copy()
        df[self.left_on] = [self.get_closest_match(x, self.left[self.left_on]) for x in df[self.right_on]]
        return self.left.merge(df, on=self.left_on, how=self.how)  # noqa: E501

    def get_closest_match(self, left: pd.Series, right: pd.Series, cutoff=cutoff) -> str or None:  # noqa: E501
        # matches = difflib.get_close_matches(left, right, n=self.n, cutoff=self.cutoff)
        matches = difflib.get_close_matches(left, right, n=self.n, cutoff=cutoff)

        return matches[0] if matches else None
