#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pipelines for feature selection based on discretized target"""


import joblib
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split, StratifiedKFold, RepeatedStratifiedKFold, RepeatedKFold, cross_val_score, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.ensemble import BaggingRegressor

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter

import utils.settings as s

s.init()
seed = s.seed
zero_loss_ratio = s.zero_loss_ratio


def ratio_multiplier(y, sampling_stategy=zero_loss_ratio):
	""""
	https://imbalanced-learn.org/stable/auto_examples/api/plot_sampling_strategy_usage.html#sphx-glr-auto-examples-api-plot-sampling-strategy-usage-py
	y : y train
	sampling_stategy = str defineing imblearn stategy or float between 0-1.0 defining ratio to which y is reduced .e.g. (0.75 >- y will be 75% of its former size)
	"""
	# if sampling_stategy == "majority": 
	#     return sampling_stategy
	# else:
	multiplier = {0: zero_loss_ratio} # set only zero loss class to the half or 3/4 of its size
	target_stats = Counter(y)
	for key, value in target_stats.items():
		if key in multiplier:
			target_stats[key] = int(value * multiplier[key])
	return target_stats


## XGBClassifier
pipe_us_xgb = Pipeline([
		('', RandomUnderSampler(
			sampling_strategy=ratio_multiplier, 
			#sampling_strategy=ratio_multiplier(sampling_stategy=zero_loss_ratio)
			), 
			#random_state=seed
		),
		('model', XGBClassifier(
			random_state=seed
		))
		])

pipe_ximput_us_xgb = Pipeline([
		('', RandomUnderSampler(
			sampling_strategy=ratio_multiplier, 
			#sampling_strategy=ratio_multiplier(sampling_stategy=zero_loss_ratio)
			), 
			#random_state=seed
		),
		('model', XGBClassifier(
			#random_state=seed
			)
		)
	]
)


## Logistic Regression
pipe_xdrop_logr = Pipeline([
		('model', LogisticRegression(
			#random_state=seed
		))
	]
)

pipe_ximput_logr = Pipeline([
		('model', LogisticRegression(
			#random_state=seed,
		))
	])

pipe_ximput_us_logr = Pipeline([
		('', RandomUnderSampler(
			sampling_strategy=ratio_multiplier
			#sampling_strategy=ratio_multiplier(sampling_stategy=zero_loss_ratio)
			), 
			#random_state=seed,
		),
		('model', LogisticRegression(
			#random_state=seed,
		))
	])

ensemble_model = {'model': BaggingRegressor,   # default bootstrap=True
		'kwargs': {'estimator': LogisticRegression()},  # TODO: pass 'random_state':seed to baggingregressor
		}
pipe_ximput_bag_logr = Pipeline([
		('', RandomUnderSampler(
			sampling_strategy=ratio_multiplier
		)),
		('bagging', ensemble_model['model'] (**ensemble_model['kwargs']) )
	])
pipe_ximput_us_bag_logr = Pipeline([
		('', RandomUnderSampler(
			sampling_strategy=ratio_multiplier
		)),
		('bagging', ensemble_model['model'] (**ensemble_model['kwargs']) )
	])


#if __name__ == "__main__":
joblib.dump(pipe_us_xgb, './pipelines/pipe_us_xgb.pkl')
joblib.dump(pipe_ximput_us_xgb, './pipelines/pipe_ximput_us_xgb.pkl')

joblib.dump(pipe_xdrop_logr, './pipelines/pipe_xdrop_logr.pkl')
joblib.dump(pipe_ximput_logr, './pipelines/pipe_ximput_logr.pkl')
joblib.dump(pipe_ximput_bag_logr, './pipelines/pipe_ximput_bag_logr.pkl')
joblib.dump(pipe_ximput_us_logr, './pipelines/pipe_ximput_us_logr.pkl')
joblib.dump(pipe_ximput_us_bag_logr, './pipelines/pipe_ximput_us_bag_logr.pkl')



