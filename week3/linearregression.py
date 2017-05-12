#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 13:34:11 2017

@author: Work
"""

# read the data and set the datetime as the index
import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/bikeshare.csv'
bikes = pd.read_csv(url, index_col='datetime', parse_dates=True)

bikes.head()

# "count" is a method, so it's best to name that column something else
bikes.rename(columns={'count':'total'}, inplace=True)

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14

# Pandas scatter plot
bikes.plot(kind='scatter', x='temp', y='total', alpha=0.2)

# Seaborn scatter plot with regression line
sns.lmplot(x='temp', y='total', data=bikes, aspect=1.5, scatter_kws={'alpha':0.2})

# create X and y
# regress 'total' on 'temp'. What are X and y?
X = bikes['temp']
y = bikes['total']
X = X.reshape(-1,1)

# import LinearRegression from sklearn, instantiate, fit X to y
from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X, y)

# print the coefficients, that is intercept_ and coef_
print(model.intercept_)
print(model.coef_)

# TODO - manually calculate the prediction
man_25 = model.intercept_ + model.coef_ * 25
print(man_25)

# TODO - use the predict method. Do they match?
print(model.predict(25))

print(man_25 - model.predict(25))

# create a new column for Fahrenheit temperature
bikes['temp_F'] = bikes.temp * 1.8 + 32
bikes.head()

# Seaborn scatter plot with regression line
sns.lmplot(x='temp_F', y='total', data=bikes, aspect=1.5, scatter_kws={'alpha':0.2})

# TODO - create X and y
X = bikes['temp_F']
X = X.reshape(-1,1)
y = bikes['total']
# TODO - instantiate and fit
modelF = LinearRegression()
modelF.fit(X, y)

# TODO - print the coefficients
print(modelF.coef_)
print(modelF.intercept_)

# convert 25 degrees Celsius to Fahrenheit
25 * 1.8 + 32
# 77
# predict rentals for 77 degrees Fahrenheit
modelF.predict(77)

"""Conclusion: The scale of the features is irrelevant for linear regression 
models. When changing the scale, we simply change our interpretation of the 
coefficients."""
# remove the temp_F column
bikes.drop('temp_F', axis=1, inplace=True)

# explore more features
feature_cols = ['temp', 'season', 'weather', 'humidity']
# multiple scatter plots in Seaborn
sns.pairplot(bikes, x_vars=feature_cols, y_vars='total', kind='reg')

# multiple scatter plots in Pandas
fig, axs = plt.subplots(1, len(feature_cols), sharey=True)
for index, feature in enumerate(feature_cols):
    bikes.plot(kind='scatter', x=feature, y='total', ax=axs[index], figsize=(16, 3))


# cross-tabulation of season and month
pd.crosstab(bikes.season, bikes.index.month)

# box plot of rentals, grouped by season
bikes.boxplot(column='total', by='season')

# line plot of rentals
bikes.total.plot()

# correlation matrix (ranges from 1 to -1)
bikes.corr()

# visualize correlation matrix in Seaborn using a heatmap
sns.heatmap(bikes.corr())

# create a list of features
feature_cols = ['temp', 'season', 'weather', 'humidity']

# create X and y
X = bikes[feature_cols]
y = bikes.total

# instantiate and fit
linreg = LinearRegression()
linreg.fit(X, y)

# print the coefficients
print linreg.intercept_
print linreg.coef_


# pair the feature names with the coefficients
zip(feature_cols, linreg.coef_)

# example true and predicted response values
true = [10, 7, 5, 5]
pred = [8, 6, 5, 10]

# calculate these metrics by hand!
from sklearn import metrics
import numpy as np
print 'MAE:', metrics.mean_absolute_error(true, pred)
print 'MSE:', metrics.mean_squared_error(true, pred)
print 'RMSE:', np.sqrt(metrics.mean_squared_error(true, pred))

# same true values as above
true = [10, 7, 5, 5]

# new set of predicted values
pred = [10, 7, 5, 13]

# MAE is the same as before
print 'MAE:', metrics.mean_absolute_error(true, pred)

# MSE and RMSE are larger than before
print 'MSE:', metrics.mean_squared_error(true, pred)
print 'RMSE:', np.sqrt(metrics.mean_squared_error(true, pred))

from sklearn.cross_validation import train_test_split

# define a function that accepts a list of features and returns testing RMSE
def train_test_rmse(feature_cols):
    X = bikes[feature_cols]
    y = bikes.total
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))

# compare different sets of features
print train_test_rmse(['temp', 'season', 'weather', 'humidity'])
print train_test_rmse(['temp', 'season', 'weather'])
print train_test_rmse(['temp', 'season', 'humidity'])

# using these as features is not allowed!
print train_test_rmse(['casual', 'registered'])

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

# create a NumPy array with the same shape as y_test
y_null = np.zeros_like(y_test, dtype=float)

# fill the array with the mean value of y_test
y_null.fill(y_test.mean())
y_null

# compute null RMSE
np.sqrt(metrics.mean_squared_error(y_test, y_null))

# create dummy variables
season_dummies = pd.get_dummies(bikes.season, prefix='season')

# print 5 random rows
season_dummies.sample(n=5, random_state=1)

# drop the first column
season_dummies.drop(season_dummies.columns[0], axis=1, inplace=True)

# print 5 random rows
season_dummies.sample(n=5, random_state=1)

# concatenate the original DataFrame and the dummy DataFrame (axis=0 means rows, axis=1 means columns)
bikes = pd.concat([bikes, season_dummies], axis=1)

# print 5 random rows
bikes.sample(n=5, random_state=1)

# include dummy variables for season in the model
feature_cols = ['temp', 'season_2', 'season_3', 'season_4', 'humidity']
X = bikes[feature_cols]
y = bikes.total
linreg = LinearRegression()
linreg.fit(X, y)
zip(feature_cols, linreg.coef_)

# compare original season variable with dummy variables
print train_test_rmse(['temp', 'season', 'humidity'])
print train_test_rmse(['temp', 'season_2', 'season_3', 'season_4', 'humidity'])

# hour as a numeric feature
bikes['hour'] = bikes.index.hour

# hour as a categorical feature
hour_dummies = pd.get_dummies(bikes.hour, prefix='hour')
hour_dummies.drop(hour_dummies.columns[0], axis=1, inplace=True)
bikes = pd.concat([bikes, hour_dummies], axis=1)

# daytime as a categorical feature
bikes['daytime'] = ((bikes.hour > 6) & (bikes.hour < 21)).astype(int)

print train_test_rmse(['hour'])
print train_test_rmse(bikes.columns[bikes.columns.str.startswith('hour_')])
print train_test_rmse(['daytime'])


