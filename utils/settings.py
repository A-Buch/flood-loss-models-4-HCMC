#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Global variables"""


def init():
    global seed, zero_loss_ratio
    seed = 42
    zero_loss_ratio = 0.5  # Fetue selection: keep only 25% of records with zero values

