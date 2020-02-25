#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 09:41:03 2020

@author: jagoodka
"""

# polynomial regression modelling

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Position_Salaries.csv')
# basically we want to build a model to check if there is a linear dependence between independent variables (X-es) and dependent variable (y)

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# first - linear regression model
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)


# second - polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)

# result of polynomial regression is fitted into the linear regression model
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualisations
# linear results
plt.scatter(X, y, c='y')
plt.plot(X, lin_reg.predict(X), c='g')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.title('Linear regression')
plt.show()

# polynomial results
#X_grid = np.arrange(min(X), max(X), 0.1)
#X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, c='y')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), c='g')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.title('Polynomial regression')
plt.show()

# predictions
# predict salary for a person with given years of experience, I need to pass a 2D array, that's why years_of_experience is placed in double square brackets
years_of_experience = [[6.5]]

# linear regression model
print(np.round(lin_reg.predict(years_of_experience), 2))

# polynomial regression model
print(np.round(lin_reg2.predict(poly_reg.fit_transform(years_of_experience)), 2))