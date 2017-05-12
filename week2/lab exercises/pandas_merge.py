#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 14:55:08 2017

@author: Work
"""

import pandas as pd
movie_url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.item'
movie_cols = ['movie_id', 'title']
movies = pd.read_table(movie_url, sep='|', header=None, names=movie_cols, usecols=[0, 1])
movies.head()

# get the size of movies dataframe (number of rows)
len(movies)

rating_url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.data'
rating_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table(rating_url, sep='\t', header=None, names=rating_cols)
ratings.head()

movie_ratings = pd.merge(movies, ratings)
# show the first five rows after merge
movie_ratings.head()
"""The display shows the first 5 rows of movie_ratings with movie_id == 1,
the first 5 movies are Toy Story, in fact there are 452 entries of Toy Story"""

# show the first 5 movies for movie_id != 1, ie not Toy Story
movie_ratings[movie_ratings["movie_id"]!=1].head()

ratings1 = movie_ratings.copy(deep = True)
ratings1 = ratings1[1:]
ratings2 = movie_ratings.copy(deep = True)
ratings[:-1]
diff = ratings2['rating'] - ratings1['rating']

print diff.head()

print movies.shape
print ratings.shape
print movie_ratings.shape

A = pd.DataFrame({'color': ['green', 'yellow', 'red'], 'num':[1, 2, 3]})
A

B = pd.DataFrame({'color': ['green', 'yellow', 'pink'], 'size':['S', 'M', 'L']})
B

# inner join only include observations in A and B
pd.merge(A, B, how='inner')

# outer join includes observations in either A or B
pd.merge(A, B, how='outer')

# left join include obs in A only
pd.merge(A, B, how='left')

# right join include obs in B only
pd.merge(A, B, how='right')
