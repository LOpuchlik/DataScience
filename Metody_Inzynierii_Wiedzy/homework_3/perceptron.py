#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:33:53 2020

@author: jagoodka
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from plotka import plot_decision_regions
from sklearn.metrics import accuracy_score


class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * \
                    (target - self.predict(xi))  # eta * (y -yi)
                self.w_[1:] += update * xi  # X
                self.w_[0] += update  # bias
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
# suma ważona

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# 3-class classifier


class Classifier:
    def __init__(self, perc1, perc2):
        self.perc1 = perc1
        self.perc2 = perc2

    def predict(self, X):
        return np.where(self.perc1.predict(X) == 1, 0, np.where(self.perc2.predict(X) == 1, 2, 1))


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y)

    y_train_perc_1 = y_train.copy()
    y_train_perc_2 = y_train.copy()

    y_train_perc_1[(y_train == 1) | (y_train == 2)] = - \
        1   # label 1, 2 are negative one
    # label 0 is positive one
    y_train_perc_1[(y_train_perc_1 == 0)] == 1

    y_train_perc_2[(y_train == 1) | (y_train == 0)] = - \
        1   # label 0 and 1 are negative one
    # label 2 is positive one
    y_train_perc_2[(y_train_perc_2 == 2)] == 1

    perc1 = Perceptron(eta=0.1)
    perc2 = Perceptron(eta=0.1, n_iter=26)
    perc1.fit(X_train, y_train_perc_1)
    perc2.fit(X_train, y_train_perc_2)

    clsif = Classifier(perc1, perc2)

    # predicted -1's and 1's
    perc1_pred = perc1.predict(X_train)
    perc2_pred = perc2.predict(X_train)

    # partial accuracies
    ac_perc1 = accuracy_score(perc1_pred, y_train_perc_1)
    ac_perc2 = accuracy_score(perc2_pred, y_train_perc_2)

    # overall accuracy
    if ac_perc1 > ac_perc2:
        y_res = np.where(perc1_pred == 1, 0, np.where(perc2_pred == 1, 2, 1))
    else:
        y_res = np.where(perc2_pred == 1, 2, np.where(perc1_pred == 1, 0, 1))

    print(f'Overall accuracy:', round(accuracy_score(y_res, y_train), 3))

    plot_decision_regions(X_train, y_train, classifier=clsif)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
