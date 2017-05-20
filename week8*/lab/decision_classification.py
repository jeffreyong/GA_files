#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 09:57:12 2017

@author: Work
"""

import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt')

df.columns = ['variance','skewness','curtosis','entropy','class']

features = ['variance','skewness', 'curtosis','entropy']

X = df[features]
Y = df['class']

import matplotlib.pyplot as plt
plt.scatter(X[features[0]],X[features[1]], c=Y, cmap='autumn')

import pandas as pd
pd.scatter_matrix(df, cmap='autumn', c=Y, diagonal='kde')

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

classifiers = {'Logistic': LogisticRegression(),
               'GaussianNB':GaussianNB(),
               'BernoulliNB': BernoulliNB(),
               'KNeighbors': KNeighborsClassifier(),
               'RandomForest': RandomForestClassifier(),
               'Decision': DecisionTreeClassifier()}

from sklearn.metrics import roc_auc_score

for name, clf in classifiers.items():
    clf.fit(X,Y)
    Y_hat = clf.predict(X)
    score = roc_auc_score(Y, Y_hat)
    print "%s had an accuracy score of %0.2f"% (name, score)
    
import numpy as np
    
def plotDecision(max_depth=None):
    plt.figure(figsize=(16,8))
    # Parameters
    n_classes = 2
    plot_colors = "bry"
    plot_step = 0.02

    for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                    [1, 2], [1, 3], [2, 3]]):
        # We only take the two corresponding features
        x = X[[features[i] for i in pair]]
        y = Y
        
        clf = DecisionTreeClassifier(max_depth=max_depth)
        clf.fit(x,y)

        # Plot the decision boundary
        plt.subplot(2, 3, pairidx + 1)
        x_min, x_max = X[features[pair[0]]].min() - 1, X[features[pair[0]]].max() + 1
        y_min, y_max = X[features[pair[1]]].min() - 1, X[features[pair[1]]].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

        plt.xlabel(features[pair[0]], size='large')
        plt.ylabel(features[pair[1]], size='large')
        plt.axis()

        # Plot the training points
        plt.scatter(X[features[pair[0]]], X[features[pair[1]]], c=Y,
                        cmap=plt.cm.Paired)

        plt.axis()
    plt.suptitle("Decision surface of a decision tree using paired features", size='x-large')
    plt.legend()
    plt.show()

    
plotDecision(max_depth=2)

classifiers2 = {'Logistic': LogisticRegression(),
               'GaussianNB':GaussianNB(),
               'BernoulliNB': BernoulliNB(),
               'KNeighbors': KNeighborsClassifier(),
               'RandomForest1': RandomForestClassifier(max_depth=1),
               'RandomForest2': RandomForestClassifier(max_depth=2),
               'RandomForest3': RandomForestClassifier(max_depth=3),
               'RandomForest4': RandomForestClassifier(max_depth=4),
               'RandomForest5': RandomForestClassifier(max_depth=5),
               'Decision1': DecisionTreeClassifier(max_depth=1),
               'Decision2': DecisionTreeClassifier(max_depth=2),
               'Decision3': DecisionTreeClassifier(max_depth=3),
               'Decision4': DecisionTreeClassifier(max_depth=4),
               'Decision5': DecisionTreeClassifier(max_depth=5)}

from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,Y)
for name, clf in classifiers2.items():
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test) #accuracy score
    #roc_auc_score - why did we do that? sklearn.metrics has many metrics!
    print "%s had an accuracy score of %0.2f"% (name, score)
    
classifier3 = {'Randomforest':RandomForestClassifier,
               'DecisionTree':DecisionTreeClassifier}

x_train, x_test, y_train, y_test = train_test_split(X,Y)
for name, model in classifier3.items():
    for i in range(1,21):
        clf=model(max_depth = i)
        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        print "%s (%s) had an accuracy score of %0.4f"% (name,i, score)
        
        

