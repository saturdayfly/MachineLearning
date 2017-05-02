# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 21:21:03 2017

@author: haofeng
"""
#%%
import numpy as np
from numpy import genfromtxt
#%%
'''
分离训练数据 特征变量x和预测变量y
'''
def data_ini(dataset):
    m, n = np.shape(dataset)
    x = np.ones((m, n))
    x[:, :-1] = dataset[:, :-1]
    y = dataset[:, -1]
    return x, y
            
def batch_gradient(x, y, coefficients, l_rate, max_iterations):
    for iterate in range(max_iterations):
        hypothesis = np.dot(x, coefficients)
        bias = hypothesis - y
        gradient = np.dot(x.transpose(), bias) / len(y)
        coefficients = coefficients - l_rate * gradient
        loss = np.dot(bias.transpose(), bias) / 2
        print(loss)
    return coefficients

def predict(x_test, coefficients):
    m, n = np.shape(x_test)
    x_comp = np.ones((m, n + 1))
    x_comp[:, :-1] = x_test
    y_pred = np.dot(x_comp, coefficients)
    return y_pred        
                               
    
#%%    
if __name__ == "__main__":

    dataset = genfromtxt("D:\\ml\\train.csv", delimiter = ',')
    x, y = data_ini(dataset)
    m, n = np.shape(dataset)
    coef = np.ones(n)
    l_rate = 0.1
    max_iterations = 5000
    coef = batch_gradient(x, y, coef, l_rate, max_iterations)
    x_test = np.array([[3.1, 5.5], [3.3, 5.9], [3.5, 6.3], [3.7, 6.7], [3.9, 7.1]])
    print(predict(x_test, coef))
    

