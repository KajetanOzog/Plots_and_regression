import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC

from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=.1, random_state=42)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

hyperparameters = [[0.02, 1.0],
                  [0.1, 1.0],
                  [1, 1.0],
                  [10, 1.0],
                  [100, 1.0],
                  [100, 10],
                  [100, 100]]
for hp in hyperparameters:
    clf = SVC(gamma=hp[0], C=hp[1], kernel='rbf')
    clf.fit(X, y)
    y_hat = clf.predict(X)
    plot_decision_regions(X, y_hat, clf)
    plt.show()
