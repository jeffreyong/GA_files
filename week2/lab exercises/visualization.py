#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 15:44:08 2017

@author: Work
"""

import pandas as pd
import matplotlib.pyplot as plt

# display plots in the notebook
#%matplotlib inline

# increase default figure and font sizes for easier viewing
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14

# read in the drinks data
drink_cols = ['country', 'beer', 'spirit', 'wine', 'liters', 'continent']
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/drinks.csv'
drinks = pd.read_csv(url, header=0, names=drink_cols, na_filter=False)

# sort the beer column and mentally split it into 3 groups
drinks['beer'].order().values

drinks.beer.plot(kind='hist', bins=15, title='Histogram of beer servings')
plt.xlabel('Beer servings')
plt.ylabel('Frequencey')

# compare with density plot (smooth version of a histogram)
drinks.beer.plot(kind='density', xlim=(0,500))

drinks[['beer', 'wine']].sort_values('beer')

# compare with scatter plot
drinks.plot(kind='scatter', x = 'beer', y = 'wine')

# add transparency
drinks.plot(kind='scatter', x = 'beer', y = 'wine', alpha=0.4)

#vary point color by spirit servings
drinks.plot(kind='scatter', x='beer', y='wine', c='spirit', colormap='Blues')

# scatter matrix of three numerical columns
pd.scatter_matrix(drinks[['beer', 'spirit', 'wine']])

# increase figure size
pd.scatter_matrix(drinks[['beer', 'spirit', 'wine']], figsize=(10, 8))

# count the number of countries in each continent
drinks.continent.value_counts()

# compare with bar graph
drinks.continent.value_counts().plot(kind='bar')

# get mean alcohol amounts by continent
drinks.groupby('continent').mean()

# side-by-side bar plots
drinks.groupby('continent').mean().plot(kind='bar')

# drop the liters column
# print drinks.groupby('continent').mean().head()
drinks.groupby('continent').mean().drop('liters', axis=1).plot(kind='bar')

# stacked bar plots
drinks.groupby('continent').mean().drop('liters', axis=1).plot(kind='bar', stacked=True)

# sort the spirit column
drinks.spirit.sort_values()
drinks.spirit.order().values

drinks.spirit.describe()

drinks.spirit.plot(kind='box')

drinks.drop('liters', axis=1).plot(kind='box')

# read in the ufo data
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/ufo.csv'
ufo = pd.read_csv(url)
ufo['Time'] = pd.to_datetime(ufo.Time)
ufo['Year'] = ufo.Time.dt.year

# count the number of ufo reports each year (and sort by year)
ufo.Year.value_counts().sort_index()

# compare with line plot
ufo.Year.value_counts().sort_index().plot()

# box plot of beer servings grouped by continent
drinks.boxplot(column='beer', by='continent')

# box plot of all numeric columns grouped by continent
drinks.boxplot(by='continent')



