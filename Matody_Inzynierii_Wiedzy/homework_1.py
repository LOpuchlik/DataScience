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

#wczytanie i wstepna obróbka danych
path ='dane9.txt'
df = pd.read_csv(path, sep=' ', header=None)
df.columns = ['In', 'Out', 'To_drop']
df.drop('To_drop', axis='columns', inplace=True)
df.sample(10)

# setting variables
X = df.iloc[:, 0].values.reshape(-1, 1)
y = df.iloc[:, 1].values

# train test splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# LINEAR
linear = LinearRegression().fit(X_train, y_train)
y_pred = linear.predict(X_test)
linear_r_sqared = r2_score(y_test, y_pred)
linear_mse = mean_squared_error(y_test, y_pred)
print('-------------- Linear --------------')
print(f'R\u00B2:', np.round(linear_r_sqared, 4))
print(f'MSE:', np.round(linear_mse, 4))
print(f'Coefficients:', linear.coef_)
print(f'Interception:', linear.intercept_)
print()


# POLYNOMIAL
# making x^1, x^2, x^3
poly_3 = PolynomialFeatures(degree=3, include_bias=False)
poly_3_X_train = poly_3.fit_transform(X_train)
poly_3_X_test = poly_3.fit_transform(X_test)
# print(poly_3_X_train)

polynomyal = LinearRegression()
polynomyal.fit(poly_3_X_train, y_train)
poly_3_prediction = polynomyal.predict(poly_3_X_test)
poly_3_r_sqared = r2_score(y_test, poly_3_prediction)
poly_3_mse = mean_squared_error(y_test, poly_3_prediction)

print('------------ 3rd degree ------------')
print(f'R\u00B2:', np.round(poly_3_r_sqared, 4))
print(f'MSE:', np.round(poly_3_mse, 4))
print(f'Coefficients:', polynomyal.coef_)
print(f'Interception:', polynomyal.intercept_)



# PLOTS
fig, ax = plt.subplots()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# raw data
plt.scatter(X, y, color='black', s=10, label='Raw data')
#linear regression
plt.plot(X_test, y_pred, color='red', label='linear' f', r\u00B2={np.round(linear_r_sqared, 4)}' f', MSE={np.round(linear_mse, 4)}')

# degree 3 regression
sort_axis_by_X = operator.itemgetter(0)
sorted_zip = sorted(zip(X_test, poly_3_prediction), key=sort_axis_by_X)
X_test, poly_3_prediction = zip(*sorted_zip)
plt.plot(X_test, poly_3_prediction, color='blue', label='3rd degree' f', r\u00B2={np.round(poly_3_r_sqared, 4)}' f', MSE={np.round(poly_3_mse, 4)}')

plt.title('Out vs. In')
plt.xlabel('In')
plt.ylabel('Out')
plt.legend(loc='upper right')
plt.show();





