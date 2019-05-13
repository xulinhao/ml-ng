'''
Computes the cost of using theta as the parameter for linear 
regression to fit the data points in x and y.

    J_theta = sum((h_theta(i) - y(i)) ^ 2)/ 2m, where 
    i = 1..m and h_theta = X(i, :) * theta
'''

import numpy as np


def compute_cost(x, y, theta):

    # ====================== YOUR CODE HERE ======================

    j = 0.
    m = len(y)

    for i in range(m):
        h = np.dot(x.iloc[i], theta)
        j = j + np.power((h - y.iloc[i]), 2)

    j = j / (2 * m)
    return j

    # =========================================================================
