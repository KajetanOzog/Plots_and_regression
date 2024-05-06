import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.show()

plt.hist(y_train)
plt.hist(y_test)
plt.show()


bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_tree))

plt.figure(figsize=(10, 6))
plot_decision_regions(X_train, y_train, clf=bag_clf, legend=2)
plt.title('Decision Boundaries for BaggingClassifier')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

plt.figure(figsize=(10, 6))
plot_decision_regions(X_train, y_train, clf=tree_clf, legend=2)
plt.title('Decision Boundaries for DecisionTreeClassifier')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=500,
    bootstrap=True,
    n_jobs=-1,
    oob_score=True)

bag_clf.fit(X_train, y_train)
print(bag_clf.oob_score_)

y_pred = bag_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))