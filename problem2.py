# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 09:04:38 2020

@author: grace
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


hightemp = np.loadtxt('hightemp.txt')
lowtemp = np.loadtxt('lowtemp.txt')
precipitation = np.loadtxt('precipitation.txt')
brooklyn = np.loadtxt('brooklyn.txt')
manhattan = np.loadtxt('manhattan.txt')
williamsburg = np.loadtxt('williamsburg.txt')
queensboro = np.loadtxt('queensboro.txt')
total = np.loadtxt('total.txt')

# x variable
avgtemp = (lowtemp + hightemp) / 2
X = avgtemp.reshape(-1,1)

# y variable
y = total.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

regr = LinearRegression(fit_intercept=True)
regr.fit(X_train,y_train)
print('Coefficients', regr.coef_)
print('Intercept', regr.intercept_)

y_pred_test = regr.predict(X_test)
print(y_pred_test.shape)
print(y_pred_test)

# The mean squared error
print(f'Mean squared error: {mean_squared_error(y_test, y_pred_test)}')
# The coefficient of determination: 1 is perfect prediction
print(f'Coefficient of determination: {r2_score(y_test, y_pred_test)}')

plt.scatter(X_train, y_train, color='black', label='Train data points')
plt.scatter(X_test, y_test, color='red', label='Test data points')
plt.plot(X_test, y_pred_test, color='blue', linewidth=1, label='Model')
plt.scatter(X_test, y_pred_test, marker='x', color='red', linewidth=3, label=((str)))
plt.legend()
plt.xlabel('Temperature (F)')
plt.ylabel('Number of vehicles')
plt.show()


