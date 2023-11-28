# Author:-Saswat Tulo
# GitHub:- https://github.com/saswattulo
#email:- saswattulo@gmail.com
#Recommended Python version 3.12.0.
# This file may contain error, it is highly appreciated to notify me.






import numpy as np


def sigmoid(z):
    g = np.zeros(z.size)

    # ===================== Your Code Here =====================
    # Instructions : Compute the sigmoid of each value of z (z can be a matrix,
    #                vector or scalar
    #
    # Hint : Do not import math
    g=1/(1+np.exp(-1*z))

    return g
