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
lin_reg.fit(X,y)


# second - polynomial regression model
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
# result of polynomial regression is fitted into the linear regression model
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)



# train test splitting

"""X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)"""