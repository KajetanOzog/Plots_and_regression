import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()


params = [(1, 0.5), (2, 0.5), (2, 1), (10, 0.5), (10, 1)]
for p in params:
    ada_clf = GradientBoostingClassifier(n_estimators=p[0], learning_rate=p[1], random_state=42)
    ada_clf.fit(X_train, y_train)
    plot_decision_regions(X_train, y_train, ada_clf)
    plt.title("For n_estimators={} and learning_rate={})".format(p[0], p[1]))
    plt.show()
    y_pred = ada_clf.predict(X_test)
    print("Accuracy score for n_estimators={} and learning_rate={}: {})"
          .format(p[0], p[1], accuracy_score(y_test, y_pred)))

print("XGB:\n\n\n")

for p in params:
    ada_clf = XGBClassifier(n_estimators=p[0], learning_rate=p[1], random_state=42)
    ada_clf.fit(X_train, y_train)
    plot_decision_regions(X_train, y_train, ada_clf)
    plt.title("For n_estimators={} and learning_rate={})".format(p[0], p[1]))
    plt.show()
    y_pred = ada_clf.predict(X_test)
    print("Accuracy score for n_estimators={} and learning_rate={}: {})"
          .format(p[0], p[1], accuracy_score(y_test, y_pred)))
