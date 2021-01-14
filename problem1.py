# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 20:26:57 2020

@author: grace
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm



hightemp = np.loadtxt('hightemp.txt')
lowtemp = np.loadtxt('lowtemp.txt')
precipitation = np.loadtxt('precipitation.txt')
brooklyn = np.loadtxt('brooklyn.txt')
manhattan = np.loadtxt('manhattan.txt')
williamsburg = np.loadtxt('williamsburg.txt')
queensboro = np.loadtxt('queensboro.txt')

num_data_points = len(manhattan)
wedthurs_brooklyn = []

#get all the data from wednesday 
for i in range(num_data_points):
    if i == 5 or i ==6:
        wedthurs_brooklyn.append(queensboro[i])
    elif ((i + 2) % 7 == 0) or ((i + 1) % 7 == 0):
        wedthurs_brooklyn.append(queensboro[i])

#convert into an array for processing into a histogram
wedthurs_brooklyn = np.array(wedthurs_brooklyn)
        
    

def get_sample(sample_size):
    return np.random.choice(wedthurs_brooklyn, size = sample_size)


# get a list of x_bars
def  repeat_experiment(n_repeats = 2000, n_samples = 1000000): 
    x_bar_list = [np.mean(get_sample(n_samples)) for i in range(n_repeats)]
    return x_bar_list

x_bar_list = repeat_experiment()

print(x_bar_list)

plt.hist(x_bar_list)
plt.xlabel('xbar, sample average')
plt.legend()

mu_pop = np.mean(wedthurs_brooklyn)
#x_bar_list2 = repeat_experiment(n_repeats=500, n_samples=1000)
mse = np.mean((x_bar_list - mu_pop)**2)
print(f'The MSE IS:{mse}')

fig = sm.qqplot(np.array(x_bar_list), line='45')
plt.show()

    