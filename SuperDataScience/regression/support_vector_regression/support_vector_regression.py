#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:06:26 2020

@author: jagoodka
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# SVR requires manual feature scaling
from sklearn.preprocessing import StandardScaler
# for X
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)
# for y
scaler_y = StandardScaler()
y = np.array(y).reshape(-1,1) # reshaping 1D to 2D array for scaling to work
y = scaler_y.fit_transform(y)

# support vector regression model fitting
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# prediction
# 6.5 has to be transformed by scaler to be at the same scale as paramerer y used for fitting
# and then it has to be inversely rescaled to obtain the predicted result in the same scale as our real observstions
y_pred = np.round(scaler_y.inverse_transform(regressor.predict(scaler_X.transform(np.array([[6.5]])))), 2)
print(y_pred)

# Visualisations
# linear results
plt.scatter(X, y, c='y')
plt.plot(X, regressor.predict(X), c='g')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.title('Support Vector Regression')
plt.show()

