import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from xgboost import XGBClassifier

dataset = np.loadtxt('adult/diabetes.txt', delimiter=",")

X = dataset[:,0:8]
Y = dataset[:,8]
print(X.shape)
print(np.mean(Y))

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

seed=123
kfold = StratifiedKFold(n_splits=5)

#svm linear
pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', LinearSVC(C=1))])
param_grid = {
            'preprocessing': [StandardScaler(), None],
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]
}

grid_1 = GridSearchCV(pipe, param_grid, cv=kfold, return_train_score=True)

grid_1.fit(X_train, y_train)


#svm rbf
pipe = Pipeline([
    ('preprocessing', StandardScaler()),
    ('classifier', SVC(kernel='rbf'))
])

param_grid = {
    'preprocessing': [StandardScaler(), None],
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100]
}
grid_2 = GridSearchCV(pipe, param_grid, cv=kfold, return_train_score=True)

grid_2.fit(X_train, y_train)


#LR
pipe = Pipeline([
    ('preprocessing', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000))
])

param_grid = {
    'preprocessing': [StandardScaler(), None],
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]
}

grid_3 = GridSearchCV(pipe, param_grid, cv=kfold, return_train_score=True)

grid_3.fit(X_train, y_train)


#KNN
pipe = Pipeline([
    ('preprocessing', StandardScaler()),
    ('classifier', KNeighborsClassifier())
])

param_grid = {
    'preprocessing': [StandardScaler(), None],
    'classifier__n_neighbors': [1, 3, 5, 7, 9]
}

grid_4 = GridSearchCV(pipe, param_grid, cv=kfold, return_train_score=True)

grid_4.fit(X_train, y_train)


#Decision trees
pipe = Pipeline([
    ('preprocessing', StandardScaler()),
    ('classifier', DecisionTreeClassifier())
])

param_grid = {
    'preprocessing': [StandardScaler(), None],
    'classifier__max_depth': [None, 5, 10, 20, 30, 50]
}

grid_5 = GridSearchCV(pipe, param_grid, cv=kfold, return_train_score=True)

grid_5.fit(X_train, y_train)

#Bagging Classifier
pipe = Pipeline([
    ('preprocessing', StandardScaler()),
    ('classifier', BaggingClassifier())
])

param_grid = {
    'preprocessing': [StandardScaler(), None],
    'classifier__n_estimators': [10, 50, 100],
    'classifier__max_samples': [0.5, 1.0],
    'classifier__max_features': [0.5, 1.0]
}

grid_6 = GridSearchCV(pipe, param_grid, cv=kfold, return_train_score=True)

grid_6.fit(X_train, y_train)


#Random Forest Classifier
pipe = Pipeline([
    ('preprocessing', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

param_grid = {
    'preprocessing': [StandardScaler(), None],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 5, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_7 = GridSearchCV(pipe, param_grid, cv=kfold, return_train_score=True)

grid_7.fit(X_train, y_train)


#Extra Tree
pipe = Pipeline([
    ('preprocessing', StandardScaler()),
    ('classifier', ExtraTreesClassifier())
])

param_grid = {
    'preprocessing': [StandardScaler(), None],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 5, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_8 = GridSearchCV(pipe, param_grid, cv=kfold, return_train_score=True)

grid_8.fit(X_train, y_train)

#Ada Boost
pipe = Pipeline([
    ('preprocessing', StandardScaler()),
    ('classifier', AdaBoostClassifier())
])


param_grid = {
    'preprocessing': [StandardScaler(), None],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__learning_rate': [0.001, 0.01, 0.1, 1.0]
}

grid_9 = GridSearchCV(pipe, param_grid, cv=kfold, return_train_score=True)

grid_9.fit(X_train, y_train)

#Gradient Boosting
pipe = Pipeline([
    ('preprocessing', StandardScaler()),
    ('classifier', GradientBoostingClassifier())
])

param_grid = {
    'preprocessing': [StandardScaler(), None],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__learning_rate': [0.001, 0.01, 0.1, 1.0],
    'classifier__max_depth': [3, 5, 7]
}

grid_10 = GridSearchCV(pipe, param_grid, cv=kfold, return_train_score=True)

grid_10.fit(X_train, y_train)

#XGBClassifier
pipe = Pipeline([
    ('preprocessing', StandardScaler()),
    ('classifier', XGBClassifier())
])

param_grid = {
    'preprocessing': [StandardScaler(), None],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__learning_rate': [0.001, 0.01, 0.1, 1.0],
    'classifier__max_depth': [3, 5, 7]
}

grid_11 = GridSearchCV(pipe, param_grid, cv=kfold, return_train_score=True)

grid_11.fit(X_train, y_train)

#voting

clf1 = LogisticRegression()
clf2 = RandomForestClassifier()
clf3 = SVC(probability=True)

voting_clf = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3)],
    voting='hard'
)

param_grid = {
    'lr__C': [0.001, 0.01, 0.1, 1, 10],
    'rf__n_estimators': [50, 100, 200],
    'svc__C': [0.001, 0.01, 0.1, 1, 10]
}

grid_12 = GridSearchCV(estimator=voting_clf, param_grid=param_grid, cv=kfold, return_train_score=True)

grid_12.fit(X_train, y_train)


models = []
models.append(('SVM linear', grid_1.best_estimator_))
models.append(('SVM rbf', grid_2.best_estimator_))
models.append(('LR', grid_3.best_estimator_))
models.append(('KNN', grid_4.best_estimator_))
models.append(('DecisionTreeClassifier', grid_5.best_estimator_))
models.append(('BaggingClassifier', grid_6.best_estimator_))
models.append(('RandomForestClassifier', grid_7.best_estimator_))
models.append(('ExtraTreesClassifier', grid_8.best_estimator_))
models.append(('AdaBoostClassifier', grid_9.best_estimator_))
models.append(('GradientBoostingClassifier', grid_10.best_estimator_))
models.append(('XGBClassifier', grid_11.best_estimator_))
models.append(('voting_clf', grid_12.best_estimator_))

precision_score = []
recall_score = []
f1_score = []
accuracy_score = []
roc_auc_score = []
for name, model in models:
    print(name)
    print("precision_score: {}".format(metrics.precision_score(y_test, model.predict(X_test))))
    print("recall_score: {}".format(metrics.recall_score(y_test, model.predict(X_test))))
    print("f1_score: {}".format(metrics.f1_score(y_test, model.predict(X_test))))
    print("accuracy_score: {}".format(metrics.accuracy_score(y_test, model.predict(X_test))))

    if (name == 'SVM linear' or name == 'SVM rbf'):
        print("roc_auc_score: {}".format(metrics.roc_auc_score(y_test, model.decision_function(X_test))))
    else:
        print("roc_auc_score: {}".format(metrics.roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])))

    precision_score.append(metrics.precision_score(y_test, model.predict(X_test)))
    recall_score.append(metrics.recall_score(y_test, model.predict(X_test)))
    f1_score.append(metrics.f1_score(y_test, model.predict(X_test)))
    accuracy_score.append(metrics.accuracy_score(y_test, model.predict(X_test)))
    if (name == 'SVM linear' or name == 'SVM rbf'):
        roc_auc_score.append(metrics.roc_auc_score(y_test, model.decision_function(X_test)))
    else:
        roc_auc_score.append(metrics.roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))


d = {'precision_score': precision_score,
     'recall_score': recall_score,
     'f1_score': f1_score,
     'accuracy_score': accuracy_score,
     'roc_auc_score': roc_auc_score
    }
df = pd.DataFrame(data=d)
df.insert(loc=0, column='Method', value=['SVM linear',
                                         'SVM rbf','LR','KNN', 'DecisionTreeClassifier','BaggingClassifier',
                                         'RandomForestClassifier','ExtraTreesClassifier', 'AdaBoostClassifier',
                                         'GradientBoostingClassifier','XGBClassifier', 'voting'])
print(df)


