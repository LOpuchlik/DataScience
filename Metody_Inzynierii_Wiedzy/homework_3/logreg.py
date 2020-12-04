#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:33:53 2020

@author: jagoodka
"""

import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from plotka import plot_decision_regions
from plotka import plot_decision_regions_part
from sklearn.metrics import accuracy_score


class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=1000, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

# 3-class classifier


class Classifier:
    def __init__(self, logreg1, logreg2):
        self.logreg1 = logreg1
        self.logreg2 = logreg2

    def predict(self, X):
        return np.where(self.logreg1.predict(X) == 1, 0, np.where(self.logreg2.predict(X) == 1, 2, 1))


def main():
    iris = datasets.load_iris()
    print(iris.feature_names)
    X = iris.data[:, [1, 2]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=1, stratify=y)

    y_train_logreg1 = y_train.copy()
    y_train_logreg2 = y_train.copy()
    X_train_logreg = X_train.copy()

    y_train_logreg1[(y_train == 1) | (y_train == 2)] = 0
    y_train_logreg1[(y_train == 0)] = 1

    y_train_logreg2[(y_train == 1) | (y_train == 0)] = 0
    y_train_logreg2[(y_train == 2)] = 1

    logreg1 = LogisticRegressionGD(eta=0.01)
    logreg2 = LogisticRegressionGD(eta=0.01)
    logreg1.fit(X_train_logreg, y_train_logreg1)
    logreg2.fit(X_train_logreg, y_train_logreg2)

    plot_decision_regions_part(
        X=X_train_logreg, y=y_train_logreg1, classifier=logreg1)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()

    plot_decision_regions_part(
        X=X_train_logreg, y=y_train_logreg2, classifier=logreg2)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()

    clsif = Classifier(logreg1, logreg2)

    # predicted -1's and 1's
    logreg1_pred = logreg1.predict(X_train)
    logreg2_pred = logreg2.predict(X_train)

    # partial accuracies
    ac_logreg1 = accuracy_score(logreg1_pred, y_train_logreg1)
    ac_logreg2 = accuracy_score(logreg2_pred, y_train_logreg2)

    # overall accuracy
    if ac_logreg1 > ac_logreg2:
        y_res = np.where(logreg1_pred == 1, 0,
                         np.where(logreg2_pred == 1, 2, 1))
    else:
        y_res = np.where(logreg2_pred == 1, 2,
                         np.where(logreg1_pred == 1, 0, 1))

    print(f'Overall accuracy:', round(accuracy_score(y_res, y_train), 3))

    plot_decision_regions(X=X_train_logreg, y=y_train, classifier=clsif)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
