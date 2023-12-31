import numpy as np
from sigmoid import *
from costFunction import *
def cost_function_reg(theta, X, y, lmd):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #
    cost,grad=cost_function(theta,X,y)
    cost=cost+(lmd/(2*m))*np.sum(theta**2)
    grad=grad+(lmd/m)*theta
    # ===========================================================

    return cost, grad
