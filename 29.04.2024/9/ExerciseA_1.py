import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve
from mlxtend.plotting import plot_decision_regions


def calculate_metrics(y_true, y_pred, y_proba):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba[:, 1])
    return precision, recall, f1, accuracy, roc_auc


def plot_decision_boundary(clf, X, y, axes=(0, 7.5, 0, 3), iris=True, legend=False, plot_training=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if not iris:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris-Setosa")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris-Versicolor")
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris-Virginica")
        plt.axis(axes)
    if iris:
        plt.xlabel("Petal length", fontsize=14)
        plt.ylabel("Petal width", fontsize=14)
    else:
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    if legend:
        plt.legend(loc="lower right", fontsize=14)


iris = load_iris()
X = iris.data[:, 2:]
y = iris.target
plt.figure(figsize=(10, 4))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

tree_clf = DecisionTreeClassifier(max_depth=10, random_state=42, criterion='entropy')
tree_clf.fit(X, y)



export_graphviz(
        tree_clf,
        out_file="./iris_tree1.dot",
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )


plt.figure(figsize=(8, 4))
plot_decision_boundary(tree_clf, X, y)
plt.plot([2.45, 2.45], [0, 3], "k-", linewidth=2)
plt.plot([2.45, 7.5], [1.75, 1.75], "k--", linewidth=2)
plt.plot([4.95, 4.95], [0, 1.75], "k:", linewidth=2)
plt.plot([4.85, 4.85], [1.75, 3], "k:", linewidth=2)
# plt.text(1.40, 1.0, "Depth=0", fontsize=15)
# plt.text(3.2, 1.80, "Depth=1", fontsize=13)
# plt.text(4.05, 0.5, "(Depth=2)", fontsize=11)
plt.show()

print(tree_clf.predict_proba([[5, 1.5]]))
print(tree_clf.predict([[5, 1.5]]))

plt.figure(figsize=(10, 5))
plot_decision_regions(X, y, tree_clf)
plt.show()


X, y = make_moons(n_samples=200, noise=.1, random_state=42)

print(X)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42, criterion='entropy')
tree_clf.fit(X, y)
export_graphviz(
        tree_clf,
        out_file="./moons_tree2.dot",
        rounded=True,
        filled=True
    )

print(tree_clf.predict_proba([[5, 1.5]]))
print(tree_clf.predict([[5, 1.5]]))

plt.figure(figsize=(10, 5))
plot_decision_regions(X, y, tree_clf)
plt.show()

deep_tree_elf2 = DecisionTreeClassifier(min_samples_leaf=4, random_state=42)
deep_tree_elf2.fit(X, y)
export_graphviz(
        tree_clf,
        out_file="./moons_tree2.dot",
        rounded=True,
        filled=True
    )

print(deep_tree_elf2.predict_proba([[5, 1.5]]))
print(deep_tree_elf2.predict([[5, 1.5]]))

plt.figure(figsize=(10, 5))
plot_decision_regions(X, y, deep_tree_elf2)
plt.show()


sample = [[5, 1.5]]
print("Probabilities Decision Tree 1:", tree_clf.predict_proba(sample))
print("Probabilities Decision Tree 2:", deep_tree_elf2.predict_proba(sample))

print("Predictions Decision Tree 1:", tree_clf.predict(sample))
print("Predictions Decision Tree 2:", deep_tree_elf2.predict(sample))





y_proba_tree1 = tree_clf.predict_proba(X)
y_pred_tree1 = tree_clf.predict(X)
precision_tree1, recall_tree1, f1_tree1, accuracy_tree1, roc_auc_tree1 = calculate_metrics(y, y_pred_tree1, y_proba_tree1)

y_proba_tree2 = deep_tree_elf2.predict_proba(X)
y_pred_tree2 = deep_tree_elf2.predict(X)
precision_tree2, recall_tree2, f1_tree2, accuracy_tree2, roc_auc_tree2 = calculate_metrics(y, y_pred_tree2, y_proba_tree2)

print("Metrics for Decision Tree 1:")
print("Precision:", precision_tree1)
print("Recall:", recall_tree1)
print("F1-score:", f1_tree1)
print("Accuracy:", accuracy_tree1)
print("ROC AUC Score:", roc_auc_tree1)

print("\nMetrics for Decision Tree 2:")
print("Precision:", precision_tree2)
print("Recall:", recall_tree2)
print("F1-score:", f1_tree2)
print("Accuracy:", accuracy_tree2)
print("ROC AUC Score:", roc_auc_tree2)

fpr_tree1, tpr_tree1, _ = roc_curve(y, y_proba_tree1[:, 1])
fpr_tree2, tpr_tree2, _ = roc_curve(y, y_proba_tree2[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr_tree1, tpr_tree1, label='Decision Tree 1 (AUC = {:.2f})'.format(roc_auc_tree1))
plt.plot(fpr_tree2, tpr_tree2, label='Decision Tree 2 (AUC = {:.2f})'.format(roc_auc_tree2))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()