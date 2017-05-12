#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 10:53:04 2017

@author: Work
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 30)

critics = pd.read_csv('https://raw.githubusercontent.com/gfleetwood/fall-2014-lessons/master/datasets/rt_critics.csv')

critics.quote[2]

critics.head()

# TODO - import both versions of naive bayes from sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

#
### How the Count Vectorizer Works
#

from sklearn.feature_extraction.text import CountVectorizer

text = ['Math is great', 'Math is really great', 'Exciting exciting Math']
print "Original text:\n\t", '\n\t'.join(text)

# TODO - create the instance of CountVectorizer class. Specify the ngram_range argument (see docs)
vectorizer = CountVectorizer(ngram_range=(1,2))
# TODO - call `fit` on the text to build the vocabulary
vectorizer.fit(text)

# display the names of the features (n grams)
vectorizer.get_feature_names()

# TODO call `transform` to convert text to a bag of words
x = vectorizer.transform(text)
print x

# CountVectorizer uses a sparse array to save memory, but it's easier in this assignment to 
# convert back to a "normal" numpy array
x_back = x.toarray()
x_back

print "Transformed text vector is \n", x

# `get_feature_names` tracks which word is associated with each column of the transformed x
print
print "Words for each feature:"
print vectorizer.get_feature_names()

# Notice that the bag of words treatment doesn't preserve information about the *order* of words, 
# just their frequency

# Instantiate the vectorizer with n-grams of length one or two
vectorizer = CountVectorizer(ngram_range=(1,2))

# Create a vector where each row is bag-of-words for a single quote
X = vectorizer.fit_transform(critics.quote) 

# TODO - Create an array where each element encodes whether the array is Fresh or Rotten
# Y = ...
# hint: apply the == condition, then use .values.astype(np.int) on the result to get the right type

Y = (critics.fresh == 'fresh').values.astype(np.int)
print Y

# Use SKLearn's train_test_split
# Important - we'll do this a thousand times
from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, Y)

# vector of all quotes
rotten_vectorizer = vectorizer.fit(critics.quote)

# a few helper functions
def accuracy_report(_clf):
    print "Accuracy: %0.2f%%" % (100 * _clf.score(xtest, ytest))

    #Print the accuracy on the test and training dataset
    training_accuracy = _clf.score(xtrain, ytrain)
    test_accuracy = _clf.score(xtest, ytest)

    print "Accuracy on training data: %0.2f" % (training_accuracy)
    
# a function to run some tests
def AnalyzeReview(testquote, _clf):
    print "\""  + testquote + "\" is judged by clasifier to be..."
    testquote = rotten_vectorizer.transform([testquote])

    if (_clf.predict(testquote)[0] == 1):
        print "... a fresh review."
    else:
        print "... a rotten review."
    return(_clf.predict(testquote)[0])

from sklearn.naive_bayes import MultinomialNB

print "MultinomialNB:"
clf_mn = MultinomialNB().fit(xtrain, ytrain)
accuracy_report(clf_mn)

from sklearn.naive_bayes import BernoulliNB
print "BernoulliNB:"
# TODO - same as above with Bernoulli
clf_b = BernoulliNB().fit(xtrain, ytrain)
accuracy_report(clf_b)

from sklearn.linear_model import LogisticRegression
print "Logistic Regression:"
# TODO - same as above with LogReg
clf_lr = LogisticRegression().fit(xtrain, ytrain)
accuracy_report(clf_lr)

AnalyzeReview("This movie was awesome", clf_mn)
AnalyzeReview("This movie was awesome", clf_b)
AnalyzeReview("This movie was awesome", clf_lr)

# Save prediction and probability

# Outputs of X (just first column)
# use clf_mn, clf_b or clf_lr
prob = clf_b.predict_proba(X)[:, 0]
predict = clf_b.predict(X)

Y==0 #(provides a mask where the actual review is bad)

# argsort returns the positions of the top n sorted values
np.argsort((prob[Y==0]))[:5]

# Top 5 Review classification errors
bad_rotten = np.argsort(prob[Y == 0])[:5]
bad_fresh = np.argsort(prob[Y == 1])[-5:]

print "Mis-predicted Rotten quotes"
print '---------------------------'
for row in bad_rotten:
    print critics[Y == 0].quote.irow(row)
    print

print "Mis-predicted Fresh quotes"
print '--------------------------'
for row in bad_fresh:
    print critics[Y == 1].quote.irow(row)
    print
    

