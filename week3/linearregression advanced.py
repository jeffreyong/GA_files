#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 15:37:01 2017

@author: Work
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import statsmodels.formula.api as smf

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14

# read data into a DataFrame
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
data.head()

data.shape

# TODO - scatter plot in Seaborn
feature_cols = ['TV', 'Radio', 'Newspaper']
# multiple scatter plots in Seaborn
sns.pairplot(data, x_vars=feature_cols, y_vars='Sales', kind='reg')

# TODO - scatter plot in Pandas
fig, axs = plt.subplots(1, len(feature_cols), sharey=True)
for index, feature in enumerate(feature_cols):
    data.plot(kind='scatter', x=feature, y='Sales', ax=axs[index], figsize=(8, 6))
    
# TODO - scatter MATRIX in Seaborn (see "pairplot")
sns.pairplot(data)

# TODO - scatter matrix in Pandas (see "scatter_matrix")
from pandas.tools.plotting import scatter_matrix
scatter_matrix(data, alpha=0.2, figsize=(10, 8), diagonal='kde')

# TODO - compute correlation matrix: call .corr()...but on what? The dataframe?
data.corr()

# TODO - display correlation matrix in Seaborn using a heatmap
# hint: call sns.heatmap(...) with what as an argument?
sns.heatmap(data.corr())

### STATSMODELS ### A different way of doing things

# create a fitted model
lm = smf.ols(formula='Sales ~ TV', data=data).fit()

# print the coefficients
lm.params

### SCIKIT-LEARN ### By now you're pros!

# TODO:
# create X (TV) and y (Sales)
X = data['TV']
X = X.values.reshape(-1,1)
y = data['Sales']

# instantiate and fit
linreg = LinearRegression()
linreg.fit(X, y)

# print the coefficients
print linreg.intercept_
print linreg.coef_

# manually calculate the prediction
7.0326 + 0.0475*50
# 9.4076

### STATSMODELS ###

# you have to create a DataFrame since the Statsmodels formula interface expects it
X_new = pd.DataFrame({'TV': [50]})

# predict for a new observation
lm.predict(X_new)

### SCIKIT-LEARN ###

# predict for a new observation
linreg.predict(50)

data['TV_dollars'] = data.TV * 1000
data.head()

### SCIKIT-LEARN ###

# create X and y
feature_cols = ['TV_dollars']
X = data[feature_cols]
y = data.Sales

# instantiate and fit
# linreg = LinearRegression()
# linreg.fit(X, y)

linreg = LinearRegression().fit(X, y)


# print the coefficients
print linreg.intercept_
print linreg.coef_

# predict for a new observation
linreg.predict(50000)

### STATSMODELS ###

# print the confidence intervals for the model coefficients
lm.conf_int()

### STATSMODELS ###

# print the p-values for the model coefficients
lm.pvalues

### STATSMODELS ###

# print the R-squared value for the model
lm.rsquared

### SCIKIT-LEARN ###

# calculate the R-squared value for the model
y_pred = linreg.predict(X)
metrics.r2_score(y, y_pred)

### SCIKIT-LEARN ###

# create X and y
feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
y = data.Sales

# instantiate and fit
linreg = LinearRegression()
linreg.fit(X, y)

# print the coefficients
print linreg.intercept_
print linreg.coef_

# pair the feature names with the coefficients
zip(feature_cols, linreg.coef_)

### STATSMODELS ###

# create a fitted model with all three features
lm = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=data).fit()

# print the p-values for the model coefficients
print lm.pvalues

# R-squared value for the model with two features
lm = smf.ols(formula='Sales ~ TV + Radio', data=data).fit()
lm.rsquared

# R-squared value for the model with three features
lm = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=data).fit()
lm.rsquared

# define a function that accepts X and y and computes testing RMSE
def train_test_rmse(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))

# include Newspaper
feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
train_test_rmse(X, y)

# exclude Newspaper
feature_cols = ['TV', 'Radio']
X = data[feature_cols]
train_test_rmse(X, y)



