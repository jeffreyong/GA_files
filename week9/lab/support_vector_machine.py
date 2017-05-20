#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 15:16:04 2017

@author: Work
"""

import numpy as np 
import matplotlib.pyplot as plt
import seaborn;
from sklearn.linear_model import LinearRegression
from scipy import stats
import pylab as pl

seaborn.set()

#Generate isotropic Gaussian blobs for clustering.
from sklearn.datasets.samples_generator import make_blobs

X, y = make_blobs(n_samples=50, centers=2,
                  random_state=0, cluster_std=0.6)
#X : array of shape [n_samples, n_features:default=2] 
#The generated samples.
#y : array of shape [n_samples]
#The integer labels for cluster membership of each sample.

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring')

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring')
xfit = np.linspace(-1, 3.5) #Generate an array if points from -1 to 3.5

# Draw three lines that couple separate the data
for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AAAAAA', alpha=0.4)

plt.xlim(-1, 3.5); #Gridbox

# import SVC
from sklearn.svm import SVC
# set clf = ... with a linear kernel
clf = SVC(kernel='linear')
# fit the data as per normal
clf.fit(X,y)

import warnings
warnings.filterwarnings('ignore')

def plot_svc_decision_function(clf, ax=None):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    x = np.linspace(plt.xlim()[0], plt.xlim()[1], 30)
    y = np.linspace(plt.ylim()[0], plt.ylim()[1], 30)
    Y, X = np.meshgrid(y, x)
    P = np.zeros_like(X)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            P[i, j] = clf.decision_function([xi, yj])
    # plot the margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

plt.scatter(X[:,0], X[:, 1], c=y, s=50, cmap='spring')
plot_svc_decision_function(clf);

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=200, facecolors='none');

from ipywidgets import interact

def plot_svm(N=100):
    X, y = make_blobs(n_samples=200, centers=2,
                      random_state=0, cluster_std=0.60)
    X = X[:N]
    y = y[:N]
    clf = SVC(kernel='linear')
    clf.fit(X, y)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring')
    plt.xlim(-1, 4)
    plt.ylim(-1, 6)
    plot_svc_decision_function(clf, plt.gca())
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=200, facecolors='none')
    
interact(plot_svm, N=[10, 200], kernel='linear');

from sklearn.datasets.samples_generator import make_circles
X1, y1 = make_circles(100, factor=.1, noise=.1)

clf = SVC(kernel='linear').fit(X1, y1)

plt.scatter(X1[:, 0], X1[:, 1], c=y1, s=50, cmap='spring')
plot_svc_decision_function(clf);

r = np.exp(-(X1[:, 0] ** 2 + X1[:, 1] ** 2))

from mpl_toolkits import mplot3d

def plot_3D(elev=30, azim=30):
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X1[:, 0], X1[:, 1], r, c=y1, s=50, cmap='spring')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')

interact(plot_3D, elev=[-90, 90], azip=(-180, 180));

clf1 = SVC(kernel='rbf')
clf1.fit(X1, y1)

plt.scatter(X1[:, 0], X1[:, 1], c=y1, s=50, cmap='spring')
plot_svc_decision_function(clf1)
plt.scatter(clf1.support_vectors_[:, 0], clf1.support_vectors_[:, 1],
            s=200, facecolors='none');

#Our favourite data
from sklearn import datasets
from sklearn.svm import LinearSVC
iris = datasets.load_iris()
iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

svc = SVC(kernel='linear')
rbf_svc = SVC(kernel='rbf', gamma=0.7)
poly_svc = SVC(kernel='poly', degree=3)
lin_svc = LinearSVC()

X




