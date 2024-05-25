import numpy as np
import sklearn.datasets
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.linear_model import Perceptron


params = [(1000, 0.05), (1000, 0.3), (200, 0.05), (200, 0.3)]
for p in params:
    X, y = sklearn.datasets.make_moons(n_samples=p[0], noise=p[1])
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.axis('equal')
    plt.show()
    per_clf = Perceptron(random_state=42)
    per_clf.fit(X, y)
    plot_decision_regions(X, y, clf=per_clf)
    plt.show()



iris = load_iris()
X = iris.data[:, (2, 3)]  # petal length, petal width
y = (iris.target == 0).astype(np.int32)
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()
per_clf = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
per_clf.fit(X, y)
y_pred = per_clf.predict([[2, 0.5]])
plot_decision_regions(X, y, clf=per_clf)
plt.show()
