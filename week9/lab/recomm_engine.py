#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 13:46:07 2017

@author: Work
"""
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/ajschumacher/gadsdata/master/user_brand.csv", header=None)

df.columns = ['user','brand']
df.head()

import numpy as np
brands, unique_brand_map = np.unique(df['brand'], return_inverse = True)
users, unique_user_map = np.unique(df['user'], return_inverse = True)

data = {}
for h in unique_user_map:
    data[h] = unique_brand_map[np.where(unique_user_map==h)].tolist()

data2 = {}
for h in data.keys():
    data2[h]=[1 if i in data[h] else 0 for i in range(len(brands))]
    
matrix = pd.DataFrame(data2)

matrix[[0,1]].head(30)

df = pd.read_csv('https://raw.githubusercontent.com/gads14-nyc/fall_2014_lessons/master/14_recommenders/brand_data_sparse.csv')

df.info()

(df.ix[:,0]+df.ix[:,1])

def similiarity(row1, row2):
    similar =(df.ix[:,row1]+df.ix[:,row2])
    number_zeros = len(np.where(similar==0)[0])
    number_ones = len(np.where(similar==1)[0])
    number_twos = len(np.where(similar==2)[0])
    return number_twos/float(number_ones+number_twos)
    #len(np.where(similar==0)[0])

similiarity(0,1)
similiarity(1,5)
similiarity(0,0)

for i in range(10):
    for j in range(10):
        print similiarity(i,j)
        
np.where(df.ix[1,:]==1)[0].tolist()

np.where(df.ix[0,:]==1)[0].tolist()

def brand_similiarity(user, brand):
    similiar_list = []
    likers = np.where(df.ix[brand,:]==1)[0].tolist()
    for liker in likers:
        similiar_list.append(similiarity(user,liker))
    return sum(similiar_list)/len(similiar_list)

brand_similiarity(user=1, brand=2)

num_brands = df.shape[0]
brand_recommendation = []
for i in range(num_brands):
    brand_recommendation.append(brand_similiarity(user=1, brand=i))
    
def brand_recommendations(user):
    num_brands = df.shape[0]
    brand_recommendation = []
    for i in range(num_brands):
        brand_recommendation.append(brand_similiarity(user=1, brand=i))
    return brand_recommendation
brand_recommendations(1)

import matplotlib.pyplot as plt
plt.plot(brand_recommendation)

top_brands = [x for x in brand_recommendation if x > .3]
print top_brands

brand_recommendation.index(top_brands[0])

np.where(df.ix[38,:]==1)[0].tolist()

brand_recommendation.index(top_brands[1])




