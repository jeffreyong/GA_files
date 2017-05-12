#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 06:35:01 2017

@author: Work
"""

# read the data
import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(url, header=None, names=col_names)

# notice that all features are continuous
pima.head()

# TODO - create X and y
# hint: drop the 'label'
X = pima.drop('label',1)
y = pima.label

# split into training and testing sets
from sklearn.cross_validation import train_test_split
# TODO - now you're good at this!
X_train, X_test, y_train, y_test = train_test_split(X, y)

# import both Multinomial and Gaussian Naive Bayes
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import metrics

# testing accuracy of Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred_class = mnb.predict(X_test)
# TODO print the accuracy_score from the metrics library
from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred_class)

# testing accuracy of Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_class = gnb.predict(X_test)
# TODO print the accuracy_score from the metrics library
from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred_class)

