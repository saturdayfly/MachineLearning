# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 16:34:05 2017

@author: haofeng
"""

#%%
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd
#import math
#%%
'''
分离训练数据 特征变量x和预测变量y
'''
def load_data(dataset):
    m, n = np.shape(dataset)
    x = np.ones((m, n))
    x[:, :-1] = dataset[:, :-1]
    y = dataset[:, -1]
    return x, y

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))
            
def batch_gradient(x, y, coefficients, l_rate, max_iterations):
    for iterate in range(max_iterations):
        hypothesis = sigmoid(np.dot(x, coefficients)) #hypothesis为label为1的概率
        bias = hypothesis - y
        gradient = np.dot(x.transpose(), bias) / len(y)
        coefficients = coefficients - l_rate * gradient
        
    return coefficients
'''
def predict(x_test, coefficients):
    m, n = np.shape(x_test)
    x_comp = np.ones((m, n + 1))
    x_comp[:, :-1] = x_test
    y_pred = sigmoid(np.dot(x_comp, coefficients))
    return y_pred        
 '''                              
def plot_bestfit(coefficients, dataset):
    x, y = load_data(dataset)
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(np.shape(dataset)[0]):
        if int(y[i]) == 1:
            xcord1.append(x[i,0])
            ycord1.append(x[i,1])
        else:
            xcord2.append(x[i,0])
            ycord2.append(x[i,1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-4.0, 4.0, 0.1)
    y = (-coefficients[2] - coefficients[0] * x) / coefficients[1]
    
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
#%%    
if __name__ == "__main__":

    dataset = []
    fr = open('classifier.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataset.append([float(lineArr[0]), float(lineArr[1]), int(lineArr[2])])
    dataset = np.asarray(dataset)
    
    x, y = load_data(dataset)
    m, n = np.shape(dataset)
    coef = np.ones(n)
    l_rate = 0.1
    max_iterations = 5000
    coef = batch_gradient(x, y, coef, l_rate, max_iterations)
    plot_bestfit(coef,dataset)
#    x_test = np.array([[3.1, 5.5], [3.3, 5.9], [3.5, 6.3], [3.7, 6.7], [3.9, 7.1]])
#    print(predict(x_test, coef))