# beer dataset
import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/beer.txt'
beer = pd.read_csv(url, sep=' ')
beer

# define X
X = beer.drop('name', axis=1)

# K-means with 3 clusters
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=1)
kmeans.fit(X)

# review the cluster labels
kmeans.labels_

# save the cluster labels and sort by cluster
beer['cluster'] = kmeans.labels_
beer.sort_values('cluster')

# review the cluster centers
kmeans.cluster_centers_

# calcute the mean of each feature of each cluster
beer.groupby('cluster').mean()

# save the dataframe of cluster centers
centers = beer.groupby('cluster').mean()

# allow plots to appear in the notebook
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

# create a colours array for plotting
import numpy as np
colors = np.array(['red', 'green', 'blue', 'yellow'])

# scatter plot of calories versus alcohol, colored by cluster (0=red, 1=green, 2=blue)
plt.scatter(beer.calories, beer.alcohol, c=colors[beer.cluster], s = 50)

# cluster centers marked by '+'
plt.scatter(centers.calories, centers.alcohol, linewidths=3, marker="+", s=300,
    c='black')

# add labels
plt.xlabel('calories')
plt.ylabel('alcohol')

# scatter plot matrix (0=red, 1=green, 2=red)
pd.scatter_matrix(X, c=colors[beer.cluster], figsize=(10,10), s=100)

# center and scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print X_scaled

# K-means with 3 clusters on scaled data
km = KMeans(n_clusters=3, random_state=1)
km.fit(X_scaled)

# save the cluster labels and sort by clusters
beer['cluster'] = km.labels_
beer.sort_values('cluster')

# review the cluster centers
beer.groupby('cluster').mean()

# scatter plot matrix of new cluster assignments
pd.scatter_matrix(X, c=colors[beer.cluster], figsize=(10,10), s=100)

"""The Silhouette Coefficient is a common metric for evaluating clustering
 "performance" in situations when the "true" cluster assignments are not known.

A Silhouette Coefficient is calculated for each observation:

SC=(b−a) / max(a,b)

    a = mean distance to all other points in its cluster
    b = mean distance to all other points in the next nearest cluster

It ranges from -1 (worst) to 1 (best). A global score is calculated by taking
 the mean score for all observations."""

# calculate SC for K=3
from sklearn import metrics
metrics.silhouette_score(X_scaled, km.labels_)

# calculate SC for K=2 through K=19
k_range = range(2, 20)
scores =[]
for k in k_range:
    km = KMeans(n_clusters=k, random_state=1)
    km.fit(X_scaled)
    scores.append(metrics.silhouette_score(X_scaled, km.labels_))

# plot the results
plt.plot(k_range, scores)
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Coefficient')
plt.grid(True)

# K-means with 4 clusters on scaled data
km = KMeans(n_clusters=4, random_state=1)
km.fit(X_scaled)
beer['cluster'] = km.labels_
beer.sort_values('cluster')

# DBScan with eps=1 and min_samples=3
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=1, min_samples=3)
db.fit(X_scaled)

db.labels_

# save the cluster labels and sort by cluster
beer['cluster'] = km.labels_
beer.sort_values('cluster')

# review the cluster centers
beer.groupby('cluster').mean()

# scatter plot matrix of DBSCAN cluster assignments
pd.scatter_matrix(X, c=colors[beer.cluster], figsize=(10,10), s=100)
