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


for (d, c) in [(1,0), (2,0), (3,0), (3,1)]:
    poly_kernel_svm_clf = SVC(kernel="poly", degree=d, coef0=c, C=1)
    poly_kernel_svm_clf.fit(X, y)
    plot_decision_regions(X, y, poly_kernel_svm_clf)
    plt.show()




