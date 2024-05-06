import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier


X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.show()
plt.hist(y_train)
plt.hist(y_test)
plt.show()

clf_1 = BaggingClassifier(DecisionTreeClassifier(max_leaf_nodes=16, random_state=42),
    n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1, random_state=42)

clf_2 = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)

clf_1.fit(X_train, y_train)
clf_2.fit(X_train, y_train)

plt.figure(figsize=(10, 6))
plot_decision_regions(X_train, y_train, clf=clf_1, legend=2)
plt.show()

plt.figure(figsize=(10, 6))
plot_decision_regions(X_train, y_train, clf=clf_2, legend=2)
plt.show()

y_pred = clf_1.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred = clf_2.predict(X_test)
print(accuracy_score(y_test, y_pred))


clf_3 = ExtraTreesClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
clf_3.fit(X_train, y_train)
plt.figure(figsize=(10, 6))
plot_decision_regions(X_train, y_train, clf=clf_3, legend=2)
plt.show()

y_pred = clf_3.predict(X_test)
print(accuracy_score(y_test, y_pred))
