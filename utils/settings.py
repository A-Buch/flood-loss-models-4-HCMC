#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Global variables and logger functions"""

__author__ = "Anna Buch, Heidelberg University"
__email__ = "a.buch@stud.uni-heidelberg.de"



import os
import logging
import functools


def init():

    global seed
    seed = 42   # use same seed for across all methods

    global color_palette_models  # color palettes for models 
    color_palette_models = {
        "cforest":  "darkblue", 
        "ElasticNet": "steelblue", 
        "XGBRegressor":  "grey", 
    }
    # global shortnames_modelnames_colors        # define modelnames and their abbreviations and colors for plotting
    # shortnames_modelnames_colors={
    #     "Conditional Random Forest": {"cforest": "darkblue"}, 
    #     "Elastic Net": {"ElasticNet":"steelblue"}, 
    #     "XGBoost": {"XGBRegressor": "grey"}, 
    # }

## decorator for logger
def decorate_init_logger(func):
    """
    Decorator for logger
    """
    @functools.wraps(func)  # preserve original func information from magic methods such as __repr__
    def wrapper(*args):
       
        # Call the wrapped function
        logger = func(*args)

        # Log file handler
        log_file = "./tst_warning_coeff.log"
        print(f"Creating log file {log_file} due to warning that regression coefficients are wrongly calculated")
        # os.path.exists(os.path.dirname(log_file))
        if not os.path.exists(log_file):             
            open(log_file, "w+").close()

        # TODO find out how to add formatter and streamhandler from wrapped func to create logger input for log_file

        return logger

    return wrapper


## decorated/wrapped func
# @decorate_init_logger # uncoment when to make decorator permanent
def init_logger(name):
    """
    Set up a logger instance
    Modified version from <christina.ludwig@uni-heidelberg.de> for SM2T project
    name (str): Name of logger 
    log_file (str): path to log file
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%m-%d-%Y %I:%M:%S",
    )
    # Add stream handler
    streamhandler = logging.StreamHandler()
    streamhandler.setLevel(logging.INFO)
    streamhandler.setFormatter(formatter)
    if not logger.handlers: 
        logger.addHandler(streamhandler)
    
    return logger