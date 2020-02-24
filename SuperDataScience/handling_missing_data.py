#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 11:55:26 2020

@author: jagoodka
"""

import pandas as pd
from sklearn.preprocessing import Imputer

dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

# filling NaNs
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


# Imputer is only suitable to fill NaNs of numerical values 

"""
dataset_cat = pd.read_csv('Data_cat.csv')

X_cat = dataset_cat.iloc[:, :-1].values
y_cat = dataset_cat.iloc[:,-1].values

imputer_cat = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer_cat = imputer_cat.fit(X[:, 0])
X[:, 0] = imputer_cat.transform(X[:, 0])

ValueError: could not convert string to float: 'France'
"""
# first, categorical values have to be transformed into numerical values and then one can fill all of the NaNs

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# labelling X
label_encoder_X = LabelEncoder()
X[:, 0] = label_encoder_X.fit_transform(X[:, 0])

# one hot encoding X
one_hot_encoder = OneHotEncoder(categorical_features = [0]) # indicated that the first column has to be one hot encoded
X = one_hot_encoder.fit_transform(X).toarray()


# labelling y --> only 2 values, so no need for OneHotEncoding
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)
