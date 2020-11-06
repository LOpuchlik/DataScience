#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 15:06:22 2020

@author: jagoodka
"""
# imports
import operator

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# reading and wrangling data
path = 'dane9.txt'
df = pd.read_csv(path, sep=' ', header=None)
df.columns = ['In', 'Out', 'To_drop']
df.drop('To_drop', axis='columns', inplace=True)
df.sample(10)

# setting variables
X = df.iloc[:, 0].values.reshape(-1, 1)
y = df.iloc[:, 1].values

# train test splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# LINEAR
linear = LinearRegression().fit(X_train, y_train)
y_pred = linear.predict(X_test)
linear_r_sqared = r2_score(y_test, y_pred)
linear_mse = mean_squared_error(y_test, y_pred)

# counting mean squared error for linear model - from training
fitted_linear = linear.intercept_ + linear.coef_*X_train
#print(f'Fitted linear:', fitted_linear)
difference_linear = y_train-fitted_linear
diffefence_sqared_linear = difference_linear**2
mean_difference_squared_linear = np.mean(diffefence_sqared_linear)

print('-------------- Linear --------------')
print('MODEL y = ax + b')
print(f'R\u00B2:', np.round(linear_r_sqared, 4))
print(f'MSE train:', np.round(mean_difference_squared_linear, 4))
print(f'MSE test:', np.round(linear_mse, 4))
print(f'Coefficient (at x (a)):', linear.coef_)
print(f'Interception (b):', linear.intercept_)
print()

# POLYNOMIAL
# making x^1, x^2, x^3
poly_3 = PolynomialFeatures(degree=3, include_bias=False)
poly_3_X_train = poly_3.fit_transform(X_train)
poly_3_X_test = poly_3.fit_transform(X_test)
# print(poly_3_X_train)

polynomial = LinearRegression()
polynomial.fit(poly_3_X_train, y_train)
poly_3_prediction = polynomial.predict(poly_3_X_test)
poly_3_r_sqared = r2_score(y_test, poly_3_prediction)
poly_3_mse = mean_squared_error(y_test, poly_3_prediction)

# counting mean squared error for polynomial model - from training
fitted_polynomial = polynomial.intercept_ + polynomial.coef_[0]*poly_3_X_train[:, 0] + polynomial.coef_[1]*poly_3_X_train[:, 1] + polynomial.coef_[2]*poly_3_X_train[:, 2]
#print(fitted_values)
#print(poly_3_X_train[:, 0])
#print(poly_3_X_train[:, 1])
#print(poly_3_X_train[:, 2])
difference_polynomial = y_train-fitted_polynomial
diffefence_sqared_polynomial = difference_polynomial**2
mean_difference_squared_polynomial = np.mean(diffefence_sqared_polynomial)

print('------------ 3rd degree ------------')
print('MODEL y = ax\u00B3 + bx\u00B2 + cx + d')
print(f'R\u00B2:', np.round(poly_3_r_sqared, 4))
print(f'MSE train::', np.round(mean_difference_squared_polynomial, 4))
print(f'MSE test:', np.round(poly_3_mse, 4))
print(f'Coefficients (at x (c), at x\u00B2 (b), at x\u00B3 (a)):', polynomial.coef_)
print(f'Interception (d):', polynomial.intercept_)


# PLOTS
fig, ax = plt.subplots()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# raw data
plt.scatter(X_train, y_train, color='green', s=10, label='Train data')
plt.scatter(X_test, y_test, color='orange', s=10, label='Test data')

# linear regression
plt.plot(X_test, y_pred, color='red',
         label='linear' f', r\u00B2={np.round(linear_r_sqared, 4)}' f', MSE test={np.round(linear_mse, 4)}')

# degree 3 regression
sort_axis_by_X = operator.itemgetter(0)
sorted_zip = sorted(zip(X_test, poly_3_prediction), key=sort_axis_by_X)
X_test, poly_3_prediction = zip(*sorted_zip)
plt.plot(X_test, poly_3_prediction, color='blue',
         label='3rd degree' f', r\u00B2={np.round(poly_3_r_sqared, 4)}' f', MSE test={np.round(poly_3_mse, 4)}')

plt.title('Out vs. In')
plt.xlabel('In')
plt.ylabel('Out')
plt.legend(loc='upper right')
plt.show()
