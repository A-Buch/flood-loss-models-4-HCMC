#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""General utility functions"""

__author__ = "Christina Ludwig, GIScience Research Group, Heidelberg University"
__email__ = "christina.ludwig@uni-heidelberg.de"


import logging
import os
import json
import pygeos
import numpy as np
import math
import subprocess


def create_subfolder(out_dir: str, name: str):
    """
    Creates a subfolder in out_dir with given name
    :param out_dir: Output directory
    :param name: Name of new subfolder
    :return:
    """
    new_subfolder = os.path.join(out_dir, name)
    os.makedirs(new_subfolder, exist_ok=True)
    return new_subfolder


def check_config(config):
    """
    Check for missing parameters in config file
    :return:
    """
    parameters = [
        "output_dir",
        "timestamp",
        "name",
        "bbox",
        "epsg",
        "cloud_coverage",
        "ndvi_year",
        "output_dir",
    ]
    for par in parameters:
        assert par in config.keys(), f"Parameter '{par}' missing in config file."


def load_config(config_file):
    """
    Load config file
    :return:
    """
    assert os.path.exists(
        config_file
    ), f"Configuration file does not exist: {os.path.abspath(config_file)}"
    with open(config_file, "r") as src:
        config = json.load(src)
    return config


