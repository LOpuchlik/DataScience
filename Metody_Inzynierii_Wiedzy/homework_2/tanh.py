import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

from sklearn.model_selection import train_test_split


# reading and wrangling data
path = 'dane9.txt'
df = pd.read_csv(path, sep=' ', header=None)
df.columns = ['In', 'Out', 'To_drop']
df.drop('To_drop', axis='columns', inplace=True)
print(df.sample(10))

print('-------------------------------')

# setting variables to row vectors
X = df.iloc[:, 0].values.reshape(-1, 1)
y = df.iloc[:, 1].values
#print(f'X.shape', X.shape, 'y.shape', y.shape)


print('-------------------------------')

# train-test splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)
X_train = X_train.T
X_test = X_test.T
print(f'X_train.shape', X_train.shape, '\nX_test.shape', X_test.shape, '\nY_train.shape', y_train.shape, '\ny_test.shape', y_test.shape)


print('-------------------------------')

hidden_nodes = 100 # number of neurons in the hidden layer
learning_rate = 0.001

W1 = np.random.rand(hidden_nodes, 1) - 0.5
W2 = np.random.rand(1, hidden_nodes) - 0.5
B1 = np.random.rand(hidden_nodes, 1) - 0.5
B2 = np.random.rand(1, 1) - 0.5

print(f'W1: ',W1.shape)
print(f'B1: ', B1.shape)
print(f'W2: ', W2.shape)
print(f'B2: ', B2.shape)

print('-------------------------------')



for epoch in range(1, 100):
    A1 = np.tanh(W1 @ X_train + B1 @ np.ones(X_train.shape))
    A2 = W2 @ A1 + B2

    E2 = y_train - A2
    E1 = np.transpose(W2) @ E2

    dW2 = learning_rate * E2 @ np.transpose(A1)
    dB2 = learning_rate * E2 @ np.transpose(np.ones(E2.shape))
    dW1 = learning_rate * (1 - A1 * A1) * E1 @ np.transpose(X_train)
    dB1 = learning_rate * (1 - A1 * A1) * E1 @ np.transpose(np.ones(X_train.shape))

    W2 = W2 + dW2
    B2 = B2 + dB2
    W1 = W1 + dW1
    B1 = B1 + dB1
    
    # draw the figure so the animations will work
    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()

    if epoch % 1 == 0:
        plt.scatter(X_train.T, y_train, c='#51adcf', label='train') 
        plt.scatter(X_test.T, y_test, c='#a5ecd7', label='test')
        plt.scatter(X_train.T, A2.T, c='r', label='model') 
        plt.xlim([-1, 6])
        plt.ylim([-1, 1])
        plt.legend()
        fig.canvas.draw()
        plt.pause(0.1)
        plt.clf()
    
'''
print(f'W1: ', W1)
print(f'B1: ', B1)
print(f'W2: ', W2)
print(f'B2: ', B2)
'''

