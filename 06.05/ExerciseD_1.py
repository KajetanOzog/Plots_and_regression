import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


#11111111111111111111
params = [(1000, 0.05), (1000, 0.3), (200, 0.05), (200, 0.3)]
for p in params:
    X, y = datasets.make_moons(n_samples=p[0], noise=p[1])
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.axis('equal')
    plt.show()

    reg = LogisticRegression()
    reg.fit(X, y)
    plot_decision_regions(X, y, clf=reg, legend=2)
    plt.show()

#222222222222222222
iris = datasets.load_iris()
list(iris.keys())

X = iris["data"][:, 0:2]
y = (iris["target"] == 2).astype(np.int32)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

reg = LogisticRegression()
reg.fit(X, y)
plot_decision_regions(X, y, clf=reg, legend=2)
plt.show()


#333333333333333333333
iris = datasets.load_iris()
list(iris.keys())

X = iris["data"][:, (2, 3)]
y = iris["target"]

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
reg.fit(X, y)
print(reg.predict(X))
plot_decision_regions(X, y, clf=reg, legend=2)
plt.show()


