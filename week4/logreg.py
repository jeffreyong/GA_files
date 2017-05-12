#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 15:25:51 2017

@author: Work
"""
# glass identification dataset
import pandas as pd
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data'
col_names = ['id','ri','na','mg','al','si','k','ca','ba','fe','glass_type']
glass = pd.read_csv(url, names=col_names, index_col='id')
glass['assorted'] = glass.glass_type.map({1:0, 2:0, 3:0, 4:0, 5:1, 6:1, 7:1})

glass.head()

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

sns.lmplot(x='al', y='ri', data=glass, ci=None)

# TODO - scatter plot using Pandas
fig, axs = plt.subplots(1, len(col_names), sharey=True)
for index, feature in enumerate(col_names):
    glass.plot(kind='scatter', x='al', y='ri', ax=axs[index], figsize=(10, 6))


# scatter plot using Matplotlib
plt.scatter(glass.al, glass.ri)

# TODO - fit a linear regression model of 'ri' on 'al'
X = glass['al']
X = X.values.reshape(-1,1)
y = glass['ri']

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X, y)

# look at the coefficients to get the equation for the line, but then how do you plot the line?
print linreg.intercept_
print linreg.coef_

# you could make predictions for arbitrary points, and then plot a line connecting them
print linreg.predict(1)
print linreg.predict(2)
print linreg.predict(3)

# or you could make predictions for all values of X, and then plot those predictions connected by a line
ri_pred = linreg.predict(X)
plt.plot(glass.al, ri_pred, color='red')

# put the plots together
plt.scatter(glass.al, glass.ri)
plt.plot(glass.al, ri_pred, color='red')

# compute prediction for al=2 using the equation
linreg.intercept_ + linreg.coef_ * 2

# compute prediction for al=2 using the predict method
linreg.predict(2)

# examine coefficient for al
zip('al', linreg.coef_)

# increasing al by 1 (so that al=3) decreases ri by 0.0025
1.51699012 - 0.0024776063874696243

# increasing al by 1 (so that al=3) decreases ri by 0.0025
1.51699012 - 0.0024776063874696243

# TODO - scatter assorted' on 'al'
plt.scatter(glass.al, glass.assorted)

# TODO - fit a linear regression model and store the predictions
feature_cols = col_names[4]
X = glass['al']
X = X.values.reshape(-1,1)
y = glass['assorted']
linreg.fit(X, y)
assorted_pred = linreg.predict(X)

# scatter plot that includes the regression line
plt.scatter(glass.al, glass.assorted)
plt.plot(glass.al, assorted_pred, color='red')

# understanding np.where
import numpy as np
nums = np.array([5, 15, 8])

# np.where returns the first value if the condition is True, and the second value if the condition is False
np.where(nums > 10, 'big', 'small')

# examine the predictions
assorted_pred[:10]

# transform predictions to 1 or 0
assorted_pred_class = np.where(assorted_pred >= 0.5, 1, 0)
assorted_pred_class

# plot the class predictions
plt.scatter(glass.al, glass.assorted)
plt.plot(glass.al, assorted_pred_class, color='red')

# add predicted class to DataFrame
glass['assorted_pred_class'] = assorted_pred_class

# sort DataFrame by al
glass.sort('al', inplace=True)

# plot the class predictions again
plt.scatter(glass.al, glass.assorted)
plt.plot(glass.al, glass.assorted_pred_class, color='red')

# fit a linear regression model and store the class predictions
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)
# TODO - define X, y, fit it
X = glass.al
X = X.values.reshape(-1,1)
y = glass.assorted
logreg.fit(X,y)
# THEN make predictions on X
assorted_pred_class = logreg.predict(X)

# print the class predictions
assorted_pred_class

# TODO - plot the class predictions (scatter then plot red line as above)
plt.scatter(glass.al, glass.assorted)
plt.plot(glass.al, assorted_pred_class, color='red')

# TODO - store the predicted probabilites of class 1
# hint: use logreg.predict_proba then index the right values
assorted_pred_prob = logreg.predict_proba(X)[:, -1]

# plot the predicted probabilities
plt.scatter(glass.al, glass.assorted)
plt.plot(glass.al, assorted_pred_prob, color='red')

# examine some example predictions
print logreg.predict_proba(1)
print logreg.predict_proba(2)
print logreg.predict_proba(3)

# create a table of probability versus odds
table = pd.DataFrame({'probability':[0.1, 0.2, 0.25, 0.5, 0.6, 0.8, 0.9]})
table['odds'] = table.probability/(1 - table.probability)
table

# exponential function: e^1
np.exp(1)

# time needed to grow 1 unit to 2.718 units
np.log(2.718)
# inverse of log function
np.log(np.exp(5))

# add log-odds to the table
table['logodds'] = np.log(table.odds)
table

# plot the predicted probabilities again
plt.scatter(glass.al, glass.assorted)
plt.plot(glass.al, assorted_pred_prob, color='red')

# TODO - compute predicted log-odds for al=2 using the equation
logodds = logreg.intercept_ + logreg.coef_ * 2
print logodds

# TODO - convert log-odds to odds
# hint: what numpy math function to use?
odds = np.exp(logodds)
odds

# convert odds to probability
prob = odds/(1 + odds)
prob

# compute predicted probability for al=2 using the predict_proba method
logreg.predict_proba(2)[:, 1]

# examine the coefficient for al
zip('al', logreg.coef_[0])

# increasing al by 1 (so that al=3) increases the log-odds by 4.18
logodds = 0.64722323 + 4.1804038614510901
odds = np.exp(logodds)
prob = odds/(1 + odds)
prob

# compute predicted probability for al=3 using the predict_proba method
logreg.predict_proba(3)[:, 1]

# examine the intercept
logreg.intercept_

# convert log-odds to probability
logodds = logreg.intercept_
odds = np.exp(logodds)
prob = odds/(1 + odds)
prob

from sklearn import metrics
preds = logreg.predict(X)
print metrics.confusion_matrix(y, preds)



