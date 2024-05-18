#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Download OSM data using ohsome API"""

__author__ = "Christina Ludwig, GIScience Research Group, Heidelberg University"
__email__ = "christina.ludwig@uni-heidelberg.de"

import os
from ohsome import OhsomeClient


def build_ohsome_filters(layer: dict):
    """
    Builds the filter string based a dictionary of tags
    :param layer: Dictionary containing tags, geoms and type information for ohsome request
    :return:
    """
    tag_filters = []
    for key, values in layer["tags"].items():
        if isinstance(values, list):
            tag_filter = "{0} in ({1})".format(key, ", ".join(values))
        elif isinstance(values, str):
            tag_filter = f"{key}={values}"
        else:
            raise TypeError(f"{type(values)} not supported for tag values")
        tag_filters.append(tag_filter)
    ohsome_filter = "({0})".format(" or ".join(tag_filters))
    if "geoms" in layer.keys():
        geom_filter = " or ".join(["geometry:{0}".format(i) for i in layer["geoms"]])
        ohsome_filter += f" and ({geom_filter})"
    if "types" in layer.keys():
        type_filter = " or ".join(["type:{0}".format(i) for i in layer["types"]])
        ohsome_filter += f" and ({type_filter})"

    return ohsome_filter


def download_features(bpoly: list, layers: list, outdir: str, timestamp: str = None):
    """
    Downloads all OSM highways for the specified timestamp and bbox
    :param bpoly: Boudning box in geographic coordinates (minx, miny, maxx, maxy)
    :param layers: List of dictionaries containing tags, geoms, types info for ohsome requests
    :param outdir: Path to directory where osm data should be stored
    :param timestamp: Date and time in ISO-format for OSM data download
    :return:
    """
    client = OhsomeClient()

    for layer in layers:
        ohsome_filter_str = build_ohsome_filters(layer)
        response = client.elements.geometry.post(
            bpolys=bpoly,
            time=timestamp,
            filter=ohsome_filter_str,
            properties="metadata,tags",
            # bpolys=bbox, time=timestamp, filter=ohsome_filter_str, properties="tags"
        )
        response.to_json(os.path.join(outdir, f"{layer['name']}.geojson"))
