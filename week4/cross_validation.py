#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 06:45:08 2017

@author: Work
"""

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# read in the iris data
iris = load_iris()

# create X (features) and y (response)
X = iris.data
y = iris.target

# TODO - use train/test
X_train, X_test, y_train, y_test = train_test_split(X, y)

# TODO - check classification accuracy of KNN with K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y) 

# y_pred = ... (on X_test)
y_pred = knn.predict(X_test)
print metrics.accuracy_score(y_test, y_pred)

# TODO - simulate splitting a dataset of 25 observations into 5 folds
from sklearn.cross_validation import KFold
kf = KFold(25, n_folds=5, shuffle=False)
# recommended to use n_folds = 10
kf = KFold(25, n_folds=10, shuffle=False)

# print the contents of each training and testing set
print '{} {:^61} {}'.format('Iteration', 'Training set observations', 'Testing set observations')
for iteration, data in enumerate(kf, start=1):
    print '{:^9} {} {:^25}'.format(iteration, data[0], data[1])
    
from sklearn.cross_validation import cross_val_score

# 10-fold cross-validation with K=5 for KNN (the n_neighbors parameter)
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print scores

# use average accuracy as an estimate of out-of-sample accuracy
print scores.mean()

# TODO (super fun) - search for an optimal value of K for KNN
k_range = range(1, 31)
k_scores = []
# fill up k_scores!
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())

print k_scores

import matplotlib.pyplot as plt
%matplotlib inline

# TODO - plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')

# what do you notice?
""" There are some sharp peaks where the accuracy is not stable, not a good K to choose
Overfitting is likely when CV Accuracty is very high
Choose k where the line is relatively stable, e.g. 5<k<10

If accuracy is not stable throughout, try increasing range or n_folds
Alternatively, look for values where the fluctuations are limited within a range of accuracy values"""

# 10-fold cross-validation with the best KNN model
knn = KNeighborsClassifier(n_neighbors=20)
print cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean()

# 10-fold cross-validation with logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
print cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean()


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# read in the advertising dataset
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)

# create a Python list of three feature names
feature_cols = ['TV', 'Radio', 'Newspaper']

# use the list to select a subset of the DataFrame (X)
X = data[feature_cols]

# select the Sales column as the response (y)
y = data.Sales

# TODO - 10-fold cross-validation with all three features, using linear regression
lm = LinearRegression()
scores = cross_val_score(lm, X, y, cv=10, scoring='neg_mean_squared_error')
print scores

# fix the sign of MSE scores
mse_scores = -scores
print mse_scores

# convert from MSE to RMSE
rmse_scores = np.sqrt(mse_scores)
print rmse_scores

# calculate the average RMSE
print rmse_scores.mean()

# 10-fold cross-validation with two features (excluding Newspaper)
feature_cols = ['TV', 'Radio']
X = data[feature_cols]
print np.sqrt(-cross_val_score(lm, X, y, cv=10, scoring='neg_mean_squared_error')).mean()
