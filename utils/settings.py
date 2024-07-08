#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Global variables and logger functions"""

__author__ = "Anna Buch, Heidelberg University"
__email__ = "a.buch@stud.uni-heidelberg.de"



import os
import logging
import functools


# global seed
seed = 42   # use same seed across all models

# src / utils paths
OUTPATH_UTILS = "./utils"
OUTPATH_PIPES = f"{OUTPATH_UTILS}/pipelines"
# log path
OUTPATH_LOGS = f"{OUTPATH_UTILS}/../../logs"
# define input data paths
INPATH_DATA = "../input_survey_data/"
# define paths for model configurations and trained models [pickle, joblib]
OUTPATH_FINALMODELS = "../models_trained/final_models/"
OUTPATH_ESTIMATORS_NCV = "../models_trained/nested_cv_models/"
# define outpath results [figures, excel files]
OUTPATH_BN = "../model_results/bayesian_network/"
OUTPATH_FEATURES = "../model_results/selected_features/"
OUTPATH_EVAL = "../model_results/models_evaluation/"  # figures of model performance and evaluation

# plot settings
plot_settings_colorpalette_models = {
    "ElasticNet": "steelblue", 
    "cforest":  "darkblue", 
    "XGBRegressor":  "grey", 
    "RandomForestRegressor": "steelblue"  # reference model
}
plot_settings_modelnames = {
    "ElasticNet": "Elastic Net", 
    "cforest":  "Conditional Random Forest", 
    "XGBRegressor":  "XGBoost", 
}

## nice feature names for the figures
feature_names_plot = {
    "Target_relative_contentloss_euro" : "rcloss",
    "Target_contentloss_euro" : "closs",
    "Target_businessreduction" : "rbred",
    "flowvelocity" : "flow velocity",
    "shp_employees": "no. employees",
    "water_depth_cm": "water depth inside",
    "emergency_measures": "emergency measures",
    'flood_experience': "flood experience", 
    'inundation_duration_h': "inundation duration", 
    'precautionary_measures_lowcost': "non-structural measures", 
    "precautionary_measures_expensive": "structural measures",
    'bage': "building age", 
    'b_area': "building area",
    'hh_monthly_income_euro': "mthly. income", 
    "shp_avgmonthly_sale_euro": "mthly. sales",
    'resilience' : "resilience",
    'contaminations': "contaminations"
}


## decorated/wrapped func
# @decorate_init_logger # uncoment when to make decorator permanent
def init_logger(name):
    """
    Set up a logger instance
    Modified version based on code from <christina.ludwig@uni-heidelberg.de> for SM2T project
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

    # Add file handler
    if not os.path.exists(OUTPATH_LOGS):
        os.makedirs(OUTPATH_LOGS)
    log_file = f"{OUTPATH_LOGS}/logs_{name}.log"
    if not os.path.exists(log_file):
            open(log_file, "w+").close()
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


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
        log_file = f"./{OUTPATH_LOGS}/warning_regression_coeffcient.log"
        print(f"Creating log file {log_file} due to warning that regression coefficients are all non significant")
        # os.path.exists(os.path.dirname(log_file))
        if not os.path.exists(log_file):
            open(log_file, "w+").close()

        return logger

    return wrapper

