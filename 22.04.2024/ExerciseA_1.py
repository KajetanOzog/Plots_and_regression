import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions


class fixed_plan(object):
    """
    """

    def __init__(self, a=0.5, b=0):
        self.a = a
        self.b = b

        self.A = -self.a
        self.B = 1
        self.C = -self.b

    def fit(self, X, y):
        return self

    def predict(self, X):
        """Return class label after unit step"""
        y_cl = np.sign(np.dot(X, np.array([self.A, self.B])) + self.C)
        y_cl[y_cl == -1] = 0
        return y_cl


a = 1
b = .5
A = -a
B = 1
C = -b
params = [(1000, .05), (1000, .3), (200, .05), (200, .3)]
for p in params:

    X, y = sklearn.datasets.make_moons(n_samples=p[0], noise=p[1])
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.axis('equal')
    plt.show()
    x = np.arange(np.min(X[:, 0]), np.max(X[:, 0]), 0.5)
    yx = a * x + b
    plt.arrow(0, b, A, B, head_width=0.05, head_length=0.1, fc='k', ec='k')
    plt.plot(x, yx)

    y_cl = np.sign(np.dot(X, np.array([A, B])) + C)
    y_cl[y_cl == -1] = 0

    plt.scatter(X[:, 0], X[:, 1], c=y_cl)
    plt.axis('equal')
    plt.show()

    classifier = fixed_plan(a=1, b=0)
    classifier.fit(X, y)
    plot_decision_regions(X, y, classifier)
    plt.show()