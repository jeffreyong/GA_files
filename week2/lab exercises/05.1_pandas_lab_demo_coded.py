#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 11:34:52 2017

@author: Work
"""

import pandas as pd
drinks = pd.read_csv('https://raw.githubusercontent.com/misrab/SG_DAT1/master/data/drinks.csv', na_filter=False)

drinks.head(8)

# summary of data
drinks.info()

# stats summary of data
drinks.describe()

beer_servings = drinks.beer_servings

# displays a dataframe of first 10 countries by beer_servings
drinks[['country', 'beer_servings']][0:10]

drinks[drinks.continent == 'NA'].sort_index(by = 'spirit_servings')

north_america = drinks[drinks.continent == 'NA']
north_america.head(5)

drinks.wine_servings[drinks.continent == 'AF'].mean()

drinks.describe()

spirits = drinks[['country', 'spirit_servings']]
spirits.sort_values(by='spirit_servings', ascending=False).head(10)

drinks.wine_servings[drinks.continent == 'SA'].mean()

drinks.groupby('continent').mean()

drinks[['wine_servings', 'continent']].groupby('continent').wine_servings.mean().sort_values(ascending=False).head(5)

drinks.groupby('continent').wine_servings.mean().sort_values(ascending=False).iteritems().next()

maxMean = drinks.groupby('continent').wine_servings.mean().sort_values(ascending=False).head(1)
maxMean