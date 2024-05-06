import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import datasets


iris = datasets.load_iris()
print(list(iris.keys()))

X = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int32)

plt.figure(figsize=(8, 6))


# all on one
for c in [1, 10, 100]:
    log_reg_1 = LogisticRegression(C=c, random_state=42)
    log_reg_1.fit(X, y)

    X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_proba_1 = log_reg_1.predict_proba(X_new)

    plt.plot(X_new, y_proba_1[:, 1], linewidth=2, label="C = {}".format(c))
    plt.plot(X_new, y_proba_1[:, 0], "--", linewidth=2, label="Not Iris-Virginica")

plt.plot(X[y == 0], y[y == 0], "bs")
plt.plot(X[y == 1], y[y == 1], "g^")

plt.xlabel("Petal width (cm)", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis((0, 3, -0.02, 1.02))
plt.show()

# 2D
X = iris["data"][:, (2, 3)]
y = (iris["target"] == 2).astype(np.int32)

plt.figure(figsize=(10, 6))  # Create a single figure to contain all plots

for c in [1, 10, 100]:
    log_reg_1 = LogisticRegression(C=c, random_state=42)
    log_reg_1.fit(X, y)

    x0, x1 = np.meshgrid(
        np.linspace(2.9, 7, 500).reshape(-1, 1),
        np.linspace(0.8, 2.7, 200).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_proba_1 = log_reg_1.predict_proba(X_new)[:, 1].reshape(x0.shape)

    plt.contour(x0, x1, y_proba_1, cmap=plt.cm.brg, alpha=0.5,
                levels=np.linspace(0, 1, 10))

    left_right = np.array([2.9, 7])
    boundary = -(log_reg_1.coef_[0][0] * left_right + log_reg_1.intercept_[0]) / log_reg_1.coef_[0][1]

    plt.plot(left_right, boundary, "k--", linewidth=2, label="C = {}".format(c))

plt.plot(X[y == 0, 0], X[y == 0, 1], "bs", label="Not Iris-Virginica")
plt.plot(X[y == 1, 0], X[y == 1, 1], "g^", label="Iris-Virginica")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.axis((2.9, 7, 0.8, 2.7))
plt.show()


