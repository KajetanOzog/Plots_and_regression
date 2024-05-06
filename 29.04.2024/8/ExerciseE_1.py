import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn import metrics


def heatmap(values, xlabel, ylabel, xticklabels, yticklabels, cmap=None, vmin=None, vmax=None, ax=None, fmt="%0.2f"):
    if ax is None:
        ax = plt.gca()
    # plot the mean cross-validation scores
    img = ax.pcolor(values, cmap=cmap, vmin=vmin, vmax=vmax)
    img.update_scalarmappable()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(xticklabels)) + .5)
    ax.set_yticks(np.arange(len(yticklabels)) + .5)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    ax.set_aspect(1)

    for p, color, value in zip(img.get_paths(), img.get_facecolors(), img.get_array().flatten()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.mean(color[:3]) > 0.5:
            c = 'k'
        else:
            c = 'w'
        ax.text(x, y, fmt % value, color=c, ha="center", va="center")
    return img


np.random.seed(1)
wine = datasets.load_wine()
X = wine.data
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

plt.hist(y_train, alpha=0.5)
plt.hist(y_test, alpha=0.5)
plt.show()
seed=123
kfold = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)

pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())])
param_grid_svm_rbf = {
            'preprocessing': [StandardScaler(), None],
            'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]
}
grid_svm_rbf = GridSearchCV(pipe, param_grid_svm_rbf, cv=kfold, return_train_score=True)
grid_svm_rbf.fit(X_train, y_train)


param_grid_svm_poly = {
    'preprocessing': [StandardScaler(), None],
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'classifier__degree': [2, 3, 4]
}
svm_poly = Pipeline([
    ('preprocessing', StandardScaler()),
    ('classifier', SVC(kernel='poly'))
])
grid_svm_poly = GridSearchCV(svm_poly, param_grid_svm_poly, cv=kfold)
grid_svm_poly.fit(X_train, y_train)

param_grid_svm_linear = {
    'preprocessing': [StandardScaler(), None],
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]
}
svm_linear = Pipeline([
    ('preprocessing', StandardScaler()),
    ('classifier', SVC(kernel='linear'))
])
grid_svm_linear = GridSearchCV(svm_linear, param_grid_svm_linear, cv=kfold)
grid_svm_linear.fit(X_train, y_train)


param_grid_logistic_regression = {
    'preprocessing': [StandardScaler(), None],
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2']
}
logistic_regression = Pipeline([
    ('preprocessing', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000))
])
grid_logistic_regression = GridSearchCV(logistic_regression, param_grid_logistic_regression, cv=5)
grid_logistic_regression.fit(X_train, y_train)


models = [('SVM rbf', grid_svm_rbf.best_estimator_), ('SVM poly', grid_svm_poly.best_estimator_),
          ('SVM linear', grid_svm_linear.best_estimator_),
          ('Logistic regression', grid_logistic_regression.best_estimator_)]

precision_score = []
recall_score = []
f1_score = []
accuracy_score = []
for name, model in models:
    print(name)
    print("R^2: {}".format(metrics.precision_score(y_test, model.predict(X_test), average='weighted')))
    print("recall_score: {}".format( metrics.recall_score(y_test, model.predict(X_test), average='weighted')))
    print("f1_score: {}".format( metrics.f1_score(y_test, model.predict(X_test), average='weighted')))
    print("accuracy_score: {}".format( metrics.accuracy_score(y_test, model.predict(X_test))))
    precision_score.append(metrics.precision_score(y_test, model.predict(X_test), average='weighted'))
    recall_score.append(metrics.recall_score(y_test, model.predict(X_test), average='weighted'))
    f1_score.append( metrics.f1_score(y_test, model.predict(X_test), average='weighted'))
    accuracy_score.append(metrics.accuracy_score(y_test, model.predict(X_test)))

methods = ['SVM rbf', 'SVM poly', 'SVM linear', 'Logistic regression']

d = {'Method': methods,
     'precision_score': precision_score,
     'recall_score': recall_score,
     'f1_score': f1_score,
     'accuracy_score': accuracy_score}
df = pd.DataFrame(data=d)
print(df)

