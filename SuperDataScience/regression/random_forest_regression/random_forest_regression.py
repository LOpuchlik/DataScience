#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:03:57 2020

@author: jagoodka
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 100, max_depth = 3, random_state = 0)
regressor.fit(X, y)

# prediction
y_pred = regressor.predict([[6.5]])
print(y_pred)

# Visualisation
# random forest regression (non-continuous model)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, c='y')
plt.plot(X_grid, regressor.predict(X_grid), c='g')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.title('RFR')
plt.show()