#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 14:22:12 2017

@author: Work
"""

from sklearn.datasets import make_blobs
import numpy as np

num_blobs = 8
X, Y = make_blobs(centers=num_blobs, cluster_std=0.5, random_state=2)

X[:10,:]
# TODO - plot the data
import matplotlib.pyplot as plt
%matplotlib inline

plt.scatter(X[:, 0], X[:, 1], marker='o')

# TODO - plot with colors!
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y)

from sklearn.cluster import KMeans
# TODO - fit, and report the y values predicted
km = KMeans(n_clusters=8, random_state=0)

y_hat = km.fit_predict(X)

print y_hat

# TODO - plot with colors, based on PREDICTIONS
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y_hat)

# TODO find the centroids: ctroids = ...
# then use this to plot them:
ctroids = km.cluster_centers_
plt.scatter(ctroids[:,0], ctroids[:,1], s=100, c=np.unique(y_hat))

# find ideal value of K using elbow method
K = range(1,20)

inertias = []
for k in K:
    # create a new KMeans object for each value of k
    kmeans = KMeans(n_clusters=k, random_state=123)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_) # inertia: Sum of distances of samples to their closest cluster center

ideal_k = num_blobs -1 # (we know in advance that this is the ideal value for k)

# plot elbow curve
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.plot(K, inertias, 'b*-')

# plot the red circle
ax.plot(K[ideal_k], inertias[ideal_k], marker='o', markersize=12, 
      markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of distances of samples to their closest cluster center')
tt = plt.title('Elbow for K-Means clustering')  

print inertias

