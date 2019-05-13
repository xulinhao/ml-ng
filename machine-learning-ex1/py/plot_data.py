'''
Plots the data points and gives the figure axes labels of population and profit.
'''

import matplotlib.pyplot as plt


def plot_data(x, y, ax):

    # ====================== YOUR CODE HERE ======================

    ax.scatter(x, y, color='r', marker='x', linewidth=0.5)

    ax.set_xlabel('Population of City in 10,000s')
    ax.set_xticks(range(4, 25, 2))

    ax.set_ylabel('Profit in $10,000s')
    ax.set_yticks(range(-5, 26, 5))

    plt.show()

    # ============================================================
