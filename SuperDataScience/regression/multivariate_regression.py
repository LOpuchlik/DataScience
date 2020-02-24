#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 20:25:10 2020

@author: jagoodka
"""

# MULTIPLE VARIABLES REGRESSION

# replacing categorical values with dummy variables and make sure to get rid of collinearity (dummy variable trap) by removing one of the dummy columns for each feature
   # a situation in which one or several independent variables in a linear regression predict another is called multicollinearity; model can't distinguish between influences from one variable and another one and therefore it will not work properly
   # action: remove one of dummy variables for each feature that was broken into dummy variables

"""
The p-value is actually the probability of getting a sample like ours, or more extreme than ours IF the null hypothesis is true (by assuming that the null hypothesis is true and then determining how “strange” our sample really is).
--> not that strange - high p-value (we do not change oyr minds about the null hypothesis)
--> strange - low p-value (we start wondering if the null hypothesis is really true and maybe change our minds about it - to reject the null hypothesis)
"""

# selecting THE RIGHT variables
# there are a few methods to do this:
"""
1. All-in - throw all variables in the model
2. Backward elimination - step by step elimination of features having highest p-value after fitting
3. Forward selection
4. Bidirectional Elimination
5. Score Comparison - all possible models - too much of work


FASTEST and very COMMON - Backward Elimination
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('50_Startups.csv')
# basically we want to build a model to check if there is a linear dependence between independent variables (X-es) and dependent variable (y)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_enc = LabelEncoder()
X[:, -1] = label_enc.fit_transform(X[:, -1])
onehot_enc = OneHotEncoder(categorical_features= [-1])
X = onehot_enc.fit_transform(X).toarray()
X = X[:, 1:] # removing one of dummy variables (it could've been performed by a library and not manually)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
