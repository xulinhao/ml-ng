'''
Updates theta by taking iter gradient steps with learning rate alpha.
'''

import numpy as np

import compute_cost


def gradient_descent(x, y, theta, alpha, iters):
    temp_theta = theta
    j_history = np.zeros(iters)

    # ====================== YOUR CODE HERE ======================

    (m, n) = x.shape
    for iter in range(iters):
        for j in range(n):
            sum_j = 0
            for i in range(m):
                h = np.dot(x.iloc[i], theta)
                sum_j = sum_j + (h - y.iloc[i]) * x.iloc[i,j]

            temp_theta[j] = theta[j] - alpha * sum_j / m

        # update theta values simultaneously
        theta = temp_theta
        j_history[iter] = compute_cost.compute_cost(x, y, theta)

    # ============================================================

    return (theta, j_history)
