#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 10:15:09 2017

@author: Work
"""

# read the iris data into a DataFrame
import pandas as pd
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv(url, header=None, names=col_names)

iris.head()

# allow plots to appear in the notebook
%matplotlib inline
import matplotlib.pyplot as plt

# increase default figure and font sizes for easier viewing
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['font.size'] = 14

# create a custom colormap
from matplotlib.colors import ListedColormap
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# map each iris species to a number
iris['species_num'] = iris.species.map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})

# create a scatter plot of PETAL LENGTH versus PETAL WIDTH and color by SPECIES
iris.plot(kind='scatter', x='petal_length', y='petal_width', c='species_num', colormap=cmap_bold)

# create a scatter plot of SEPAL LENGTH versus SEPAL WIDTH and color by SPECIES
iris.plot(kind='scatter', x='sepal_length', y='sepal_width', c='species_num', colormap=cmap_bold)

iris.head()

# store feature matrix in "X"
feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = iris[feature_cols]

# alternative ways to create "X"
X = iris.drop(['species', 'species_num'], axis=1)
X = iris.loc[:, 'sepal_length':'petal_width']
X = iris.iloc[:, 0:4]

# store response vector in "y"
y = iris['species_num']

# check X's type
print type(X)
print type(X.values)

# check y's type
print type(y)
print type(y.values)

# check X's shape (n = number of observations, p = number of features)
print X.shape

# check y's shape (single dimension with length n)
print y.shape

# import KNeighborsClassifier from sklearn.
# where is it? Google the documentation
from sklearn.neighbors import KNeighborsClassifier

# make an instance of a KNeighborsClassifier object
knn = KNeighborsClassifier(n_neighbors=1)
type(knn)

print knn

# fit the knn model. What might the function be called? Documentation...
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y) 

# make predictions on this input: [3, 5, 4, 2]
# Again, what might the prediction function be called for knn?
X1 = [[3, 5, 4, 2]]
print(knn.predict(X1))

# now make predictions for [3, 5, 4, 2], [5, 4, 3, 2]
X2 = [[3, 5, 4, 2], [5, 4, 3, 2]]
print(knn.predict(X2))

# confirm prediction is an numpy array
print type(knn.predict(X))

# instantiate the model (using the value K=5)
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')

# fit the model with data
knn.fit(X, y)

X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
# predict the response for new observations
print(knn.predict(X_new))

# calculate predicted probabilities of class membership
knn.predict_proba(X_new)

print(knn.predict([[5.0, 3.6, 1.4, 0.2]]))
knn.predict_proba([[5.0, 3.6, 1.4, 0.2]])


