'''
Machine Learning Online Class - Exercise 1: Linear Regression

    Data Schema
    ------------
    x refers to the population size in 10,000s
    y refers to the profit in $10,000s
'''

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from matplotlib import interactive
from mpl_toolkits.mplot3d import Axes3D

import plot_data
import compute_cost
import gradient_descent


def pause():
    input('Press enter to continue.\n')


def plot(ax):
    print('Plotting data ...\n')

    par_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    ex1_path = os.path.abspath(os.path.join(par_path, 'ex1'))

    data = pd.read_csv(os.path.join(ex1_path, 'ex1data1.txt'), 
        names=['x1', 'y'], header=None)
    plot_data.plot_data(data['x1'], data['y'], ax)

    # make the matrix X and the result vector y
    y = data['y']
    x = pd.DataFrame(np.ones(len(y)), columns=['x0'])
    x['x1'] = data['x1']

    return (x, y)


def train_model(x, y, ax):
    # gradient descent settings
    (_, n) = x.shape
    iters = 1500
    alpha = 0.01
    theta = np.zeros(n)

    # compute and display initial cost
    print('Testing the cost function ...\n')
    j = compute_cost.compute_cost(x, y, theta)

    print('  With theta = [0.0, 0.0]')
    print('  Cost computed = %0.2f' % j)
    print('  Expected cost value (approx) 32.07\n')

    # run gradient descent
    print('Running Gradient Descent ...\n')
    (theta, j_history) = gradient_descent.gradient_descent(x, y,
        theta, alpha, iters)

    print('  Theta found by gradient descent:')
    print('  ', theta)
    print('  Expected theta values (approx):')
    print('  [-3.6303, 1.1664]\n')

    return (alpha, theta, j_history)


def plot_model(x, theta, ax):
    ax.plot(x['x1'], np.dot(x, theta), '-')
    ax.legend(['Training data', 'Linear regression'])
    ax.set_title('Training data with linear regression fit')

    ax.set_xticks(range(4, 25, 2))
    ax.set_yticks(range(-5, 26, 5))

    plt.show()


def plot_learning_history(alpha, j_history, ax):
    m = len(j_history)
    ax.plot(j_history, '-')
    ax.set_xlim(0, m)
    ax.set_title('Convergence of gradient descent with learning rate=%.2f' 
        % alpha)

    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Cost J($\\theta$)')

    ax.set_xticks(range(0, m+1, 300))
    ax.set_yticks(range(0, 10, 2))

    plt.show()


def predict(theta):
    p1 = np.dot([1, 3.5], theta) * 10000
    print('For population=35,000, we predict a profit of %0.4f' % p1)
    
    p2 = np.dot([1, 7.0], theta) * 10000
    print('For population=70,000, we predict a profit of %0.4f\n' % p2)


def plot_j_history(x, y, theta, ax3, ax4):
    print('Visualizing J(theta_0, theta_1) ...\n')

    # grid over which we will calculate j_vals
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    # calculate j_vals
    j_vals = np.zeros([len(theta0_vals), len(theta1_vals)])
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            t = [theta0_vals[i], theta1_vals[j]]
            j_vals[i,j] = compute_cost.compute_cost(x, y, t)

    # make x, y and z data objects
    axis_z = np.transpose(j_vals)
    axis_x, axis_y = np.meshgrid(theta0_vals, theta1_vals)

    # plot a new 3d surface figure
    surf = ax3.plot_surface(axis_x, axis_y, axis_z, rstride=1, cstride=1, 
        cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax3.get_figure().colorbar(surf, shrink=0.5, aspect=10)
    ax3.set_title('Surface')

    ax3.set_xlabel('$\\theta_0$')
    ax3.set_ylabel('$\\theta_1$')

    ax3.set_xticks(range(-10, 11, 5))
    ax3.set_yticks(range(-1, 5, 1))

    plt.show()

    # plot the corresponding contour figure
    cs = ax4.contour(axis_x, axis_y, np.log10(axis_z))
    ax4.plot(theta[0], theta[1], color='r', marker='x', linewidth=0.5)
    ax4.set_title('Contour, showing minimum')

    ax4.set_xlabel('$\\theta_0$')
    ax4.set_ylabel('$\\theta_1$')

    ax4.set_xticks(range(-10, 11, 2))
    ax4.set_yticks(np.linspace(-1, 4, 11))

    plt.show()


def main():
    interactive(True) # enable interactive figure display
    
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax4 = fig.add_subplot(2, 2, 4)

    (x, y) = plot(ax1)
    pause()

    (alpha, theta, j_history) = train_model(x, y, ax1)
    plot_model(x, theta, ax1)
    plot_learning_history(alpha, j_history, ax2)
    pause()

    predict(theta)
    pause()

    plot_j_history(x, y, theta, ax3, ax4)
    pause()


if __name__ == '__main__':
    main()
