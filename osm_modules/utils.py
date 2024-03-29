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


def init_logger(name, log_file_name=None):
    """
    Set up a logger instance with stream and file logger
    :return:
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
    if log_file_name:
        assert os.path.exists(
            os.path.dirname(log_file_name)
        ), "Error during logger setup: Directory of log file does not exist."
        filehandler = logging.FileHandler(filename=log_file_name)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    return logger


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


def calc_compactness(features):
    """
    Calculate compactness of a polygon
    :return:
    """
    pygeos_geoms = features.apply(
        lambda x: pygeos.from_shapely(x["geometry"]), axis=1
    ).values
    area = pygeos.area(pygeos_geoms)
    perimeter = pygeos.length(pygeos_geoms)
    return (area * 4 * np.pi) / (perimeter * perimeter)


def split_bbox(bbox, tile_size):
    """
    Split bounding box in tiles

    :param bbox: bounding box in format [xmin, ymin, xmax, ymax]
    :param tile_size: the size of the tiles in x and y direction in crs coordinates in format (x_tilesize, y_tilesize)
    :return:
    """
    x_min, y_min, x_max, y_max = bbox
    dx, dy = tile_size

    # Number of full tiles in x and y direction
    x_tiles = math.floor(round(x_max - x_min, 6) / dx)
    y_tiles = math.floor(round(y_max - y_min, 6) / dy)

    # Remainder of bbox in x and y direction
    x_rest = round(x_max - x_min, 6) % dx
    y_rest = round(y_max - y_min, 6) % dy

    for y in range(0, y_tiles):
        for x in range(0, x_tiles):
            yield tuple(
                np.array(
                    [
                        x_min + dx * x,
                        y_min + dy * y,
                        x_min + dx * (x + 1),
                        y_min + dy * (y + 1),
                    ]
                ).round(6)
            )
        if x_rest != 0:
            yield tuple(
                np.array(
                    [x_min + dx * (x + 1), y_min + dy * y, x_max, y_min + dy * (y + 1)]
                ).round(6)
            )

    # Last row
    if y_rest != 0:
        for x in range(0, x_tiles):
            yield tuple(
                np.array(
                    [x_min + dx * x, y_min + dy * y_tiles, x_min + dx * (x + 1), y_max]
                ).round(6)
            )
        yield tuple(
            np.array([x_min + dx * x_tiles, y_min + dy * y_tiles, x_max, y_max]).round(
                6
            )
        )


def rasterize(lu_polygons_file, column, epsg=None):
    """
    Rasterize a polygons file
    :param polygons_file:
    :return:
    """
    lu_polygons_tif_file = os.path.splitext(lu_polygons_file)[0] + ".tif"
    lu_polygons_tif_temp = os.path.splitext(lu_polygons_file)[0] + "_tmp.tif"
    cmd = [
        "gdal_rasterize",
        "-a",
        column,
        "-of",
        "GTiff",
        "-tr",
        "1",
        "1",
        "-ot",
        "Float32",
        lu_polygons_file,
        lu_polygons_tif_temp,
    ]
    subprocess.call(cmd)

    if epsg:
        cmd = [
            "gdalwarp",
            "-t_srs",
            "EPSG:" + epsg,
            lu_polygons_tif_temp,
            lu_polygons_tif_file,
        ]
        subprocess.call(cmd)
        os.unlink(lu_polygons_tif_temp)
    else:
        os.rename(lu_polygons_tif_temp, lu_polygons_tif_file)
    return lu_polygons_tif_file
