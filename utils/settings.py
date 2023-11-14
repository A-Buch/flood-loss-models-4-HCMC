#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Global variables and logger functions"""

__author__ = "Anna Buch, Heidelberg University"
__email__ = "a.buch@stud.uni-heidelberg.de"



import os
import logging


def init():
    global seed
    seed = 42   # use same seed for across all methods


def init_logger(name, log_file=None):
    """
    Set up a logger instance with stream and file logger. Modified version from Christina Ludwigs version for SM2T
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
    logger.addHandler(streamhandler)
    # Log file handler
    if log_file:
        assert os.path.exists(
            os.path.dirname(log_file)
        ), "Error during logger setup: Directory of log file does not exist."
        filehandler = logging.FileHandler(filename=log_file)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    return logger