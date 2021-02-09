#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 20:51:19 2020

@author: jagoodka
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:, 2:-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# only needed or faster plotting
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)


# fitting
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=10, max_depth=5, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)


# prediction
y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)


# Visualisations

def conf_matrix_visual(X, y, plot_name):
    from matplotlib.colors import ListedColormap
    X_set, y_set = X, y
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.5, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title(plot_name)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.savefig(plot_name) # saving a plot as png in current working directory - name of a file given as the third parameter
    plt.show()


# train set results
conf_matrix_visual(X_train, y_train, 'Training_set_Random_forest_classifier')


# test set results
conf_matrix_visual(X_test, y_test,'Test_set_Random_forest_classifier')

