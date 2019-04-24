#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:25:52 2019

@author: dariolunni
"""

import pandas as pd


# Read the data
data = pd.read_csv('home-data-for-ml-course/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

from xgboost import XGBRegressor

my_model = XGBRegressor()
my_model.fit(X_train, y_train)