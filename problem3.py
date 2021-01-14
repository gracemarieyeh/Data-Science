# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 09:44:56 2020

@author: grace
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm


hightemp = np.loadtxt('hightemp.txt')
lowtemp = np.loadtxt('lowtemp.txt')
precipitation = np.loadtxt('precipitation.txt')
brooklyn = np.loadtxt('brooklyn.txt')
manhattan = np.loadtxt('manhattan.txt')
williamsburg = np.loadtxt('williamsburg.txt')
queensboro = np.loadtxt('queensboro.txt')
total = np.loadtxt('total.txt')

#separate the days which rained and days which did not rain
raintraffic = []
noraintraffic = []

for i, value in enumerate(precipitation):
    if value != 0:
        raintraffic.append(total[i])
    else:
        noraintraffic.append(total[i])


        
#sketch a histogram of probabilities  curve to see if there is significant difference

plt.xlabel('Number of Bikes')
plt.ylabel('Probability (x10^-5)')
plt.hist(noraintraffic, density=1)


#plt.hist(noraintraffic)



fig = sm.qqplot(np.array(noraintraffic), line='45')
plt.show()






