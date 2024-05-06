import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import make_moons


X, y = make_moons(n_samples=200, noise=.1, random_state=42)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()


c_vals = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

for current_c in c_vals:
    svc = SVC(kernel="linear", C=current_c)
    svc.fit(X, y)
    plot_decision_regions(X,y,svc)
    plt.show()