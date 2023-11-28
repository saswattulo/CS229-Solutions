# Author:-Saswat Tulo
# GitHub:- https://github.com/saswattulo
#email:- saswattulo@gmail.com
#Recommended Python version 3.12.0.
# This file may contain error, it is highly appreciated to notify me.





import numpy as np
from sigmoid import *


def cost_function(theta, X, y):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #

    res = sigmoid(np.dot(X, theta))

    cost = np.sum(-y * np.log(res) - (1 - y) * np.log(1 - res)) / m
    grad = np.dot(X.T, (res - y)) / m

    # ===========================================================

    return cost, grad

