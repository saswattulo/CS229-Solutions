# Author:-Saswat Tulo
# GitHub:- https://github.com/saswattulo
#email:- saswattulo@gmail.com
#Recommended Python version 3.12.0.
# This file may contain error, it is highly appreciated to notify me.





import numpy as np
from sigmoid import *


def predict(theta, X):
    m = X.shape[0]

    # Return the following variable correctly
    p = np.zeros(m)

    # ===================== Your Code Here =====================
    # Instructions : Complete the following code to make predictions using
    #                your learned logistic regression parameters.
    #                You should set p to a 1D-array of 0's and 1's
    #

    p=sigmoid(X.dot(theta.T))
    # ===========================================================

    return p
