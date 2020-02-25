#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 19:02:29 2020

@author: jagoodka
"""
#simple linear regression model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Salary_Data.csv')

# Salary is predicted based on years of experience feature
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# in simple linear regression there's no need to scale features cause the library takes care of it

""" Fitting simple regression model"""
from sklearn.linear_model import LinearRegression

# fitting simple linear regressor to training data
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting results based on testing data
# y_test - contains the real values of salaries
# y_pred - contains predicted values of salaries
y_pred = regressor.predict(X_test)

# plotting
# train set results
plt.scatter(X_train, y_train, c='y')
plt.plot(X_train, regressor.predict(X_train), c='g')

plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.title('Training set and regression line')
plt.show()

# test set results
plt.scatter(X_test, y_test, c='y')
plt.plot(X_train, regressor.predict(X_train), c='g')

plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.title('Test set and regression line')
plt.show()

