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

# Choosing right features - building an optimal model by using backward elimination

import statsmodels.formula.api as sm
# add column of 1 here representing the free coefficient in regression equation

X = np. append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

# backward elimination
# X_opt - includes only all the variables that are statistically significant
X_opt = X[:, [0,1,2,3,4,5]]
# regressor_OLS ordinary least squares
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#final set of features chosen - actually - one variable and the free parameter
X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


# ADDITIONAL STUFF
# automated backward elimination with parameters - initial X set with all features and significancy level
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x

































