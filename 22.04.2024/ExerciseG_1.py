import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression


wine = load_wine()


wine.target[wine.target == 0] = 1


X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, stratify=wine.target, random_state=42)


C_values = [1, 10, 0.01]


feature_index = 0


plt.figure(figsize=(10, 6))

for C in C_values:
    logistic_regression_model = LogisticRegression(C=C, random_state=42)

    logistic_regression_model.fit(X_train[:, [feature_index]], y_train)

    X_new = np.linspace(np.min(X_train[:, feature_index]), np.max(X_train[:, feature_index]), 100).reshape(-1, 1)

    y_proba = logistic_regression_model.predict_proba(X_new)

    plt.plot(X_new, y_proba[:, 1], linewidth=2, label="C = {}".format(C))
    plt.plot(X_new, y_proba[:, 0], "--", linewidth=2)

plt.xlabel(wine.feature_names[feature_index], fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.title("Probability of Class 1 vs. {}".format(wine.feature_names[feature_index]), fontsize=16)
plt.grid(True)
plt.show()

