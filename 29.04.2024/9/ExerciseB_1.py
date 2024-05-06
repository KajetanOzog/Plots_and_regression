import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


np.random.seed(1)
cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

plt.hist(y_train, alpha=0.5)
plt.hist(y_test, alpha=0.5)
plt.show()
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

plt.figure(figsize=(15, 10))
plot_tree(tree, filled=True, feature_names=cancer.feature_names, class_names=cancer.target_names)
plt.show()


importances = tree.feature_importances_
feature_labels = cancer.feature_names
print(feature_labels)
plt.figure(figsize=(10, 6))
plt.barh(range(X.shape[1]), importances, align='center', color='skyblue')
plt.yticks(np.arange(X.shape[1]), feature_labels)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.show()

np.random.seed(1)
wine = datasets.load_wine()

X = wine.data
y = wine.target
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X, y)

plt.figure(figsize=(15, 10))
plot_tree(tree, filled=True, feature_names=wine.feature_names, class_names=wine.target_names)
plt.show()
importances = tree.feature_importances_
feature_labels = wine.feature_names
print(feature_labels)
plt.figure(figsize=(10, 6))
plt.barh(range(X.shape[1]), importances, align='center', color='skyblue')
plt.yticks(np.arange(X.shape[1]), feature_labels)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.show()