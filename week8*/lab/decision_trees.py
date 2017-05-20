#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 09:57:12 2017

@author: Work
"""

# vehicle data
import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/vehicles_train.csv'
train = pd.read_csv(url)

# before splitting anything, just predict the mean of the entire dataset
train['prediction'] = train.price.mean()
train

# TODO - calculate RMSE for those predictions
# try using the metrics built-in feature for SSE
from sklearn import metrics

metrics.mean_squared_error('price', 'prediction')


train['price']

len('price')
len('prediction')

