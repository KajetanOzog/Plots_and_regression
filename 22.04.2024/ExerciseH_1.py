import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from matplotlib.colors import ListedColormap
from mlxtend.plotting import plot_decision_regions


iris = datasets.load_iris()

X = iris["data"][:, (2, 3)]
y = iris["target"]+1

plt.figure(figsize=(10, 4))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()


softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10, random_state=42) # "ovr"
softmax_reg.fit(X, y)

x0, x1 = np.meshgrid(
        np.linspace(0, 8, 500).reshape(-1, 1),
        np.linspace(0, 3.5, 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]


y_proba = softmax_reg.predict_proba(X_new)
y_predict = softmax_reg.predict(X_new)

zz1 = y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X[y == 3, 0], X[y == 3, 1], "g^", label="Iris-Virginica")
plt.plot(X[y == 2, 0], X[y == 2, 1], "bs", label="Iris-Versicolor")
plt.plot(X[y == 1, 0], X[y == 1, 1], "yo", label="Iris-Setosa")

custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

plt.contourf(x0, x1, zz, cmap=custom_cmap)
contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.show()

plt.figure(figsize=(10, 4))
plot_decision_regions(X, y, clf=softmax_reg)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

# one vs all
softmax_reg_ovr = LogisticRegression(multi_class="ovr", solver="lbfgs", C=10, random_state=42)
softmax_reg_ovr.fit(X, y)

# one vs one
softmax_reg_ovo = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10, random_state=42)
softmax_reg_ovo.fit(X, y)

# one vs all
plt.figure(figsize=(10, 4))
plot_decision_regions(X, y, clf=softmax_reg_ovr)
plt.xlabel('Petal length', fontsize=14)
plt.ylabel('Petal width', fontsize=14)
plt.title('Decision Regions - One-vs-All', fontsize=16)
plt.legend(loc='upper left')
plt.show()

# one vs one
plt.figure(figsize=(10, 4))
plot_decision_regions(X, y, clf=softmax_reg_ovo)
plt.xlabel('Petal length', fontsize=14)
plt.ylabel('Petal width', fontsize=14)
plt.title('Decision Regions - One-vs-One', fontsize=16)
plt.legend(loc='upper left')
plt.show()