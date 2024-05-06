from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]


class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X], index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                   na_values=[" ?"],
                   header=None, index_col=False,
                   names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                          'marital-status', 'occupation', 'relationship', 'race', 'gender',
                          'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                          'income'])

# For illustration purposes, we only select some of the columns
data = data[['workclass', 'age', 'education', 'education-num', 'occupation', 'capital-gain','gender', 'hours-per-week',  'income']]

# Setting a value to np.nan instead of None
data.at[0, 'education-num'] = np.nan

print(data.head())

data = data[1:1000]
print(data.isnull().sum())
print(data["workclass"].value_counts())
print(data["education"].value_counts())
print(data["gender"].value_counts())
print(data["occupation"].value_counts())

X = data.drop(['income'], axis=1)
y = data['income'].values
y[y == ' <=50K'] = 0
y[y == ' >50K'] = 1
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("X.shape: {} y.shape: {}".format(X.shape, y.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


imputer = SimpleImputer(strategy="median")

num_pipeline = Pipeline([
        ("select_numeric", DataFrameSelector(["education-num"])),
        ("imputer", SimpleImputer(strategy="median")),])

print(num_pipeline.fit_transform(X_train))


cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["workclass", "education", "occupation", "gender"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse_output=False, handle_unknown = 'ignore')),
    ])

print(cat_pipeline.fit_transform(X_train))

preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

print(preprocess_pipeline.fit_transform(X_train))


seed=123
kfold = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
pipe_linear = Pipeline([
    ('preprocessing', preprocess_pipeline),
    ('classifier', SVC(kernel='linear'))])

param_grid = {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]
}

grid_linear = GridSearchCV(pipe_linear, param_grid, cv=kfold)
grid_linear.fit(X_train, y_train)


pipe_rbf = Pipeline([
    ('preprocessing', preprocess_pipeline),
    ('classifier', SVC())])

param_grid = {
            'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]
}
grid_rbf = GridSearchCV(pipe_rbf, param_grid, cv=kfold)
grid_rbf.fit(X_train, y_train)


pipe_poly = Pipeline([
    ('preprocessing', preprocess_pipeline),
    ('classifier', SVC(kernel='poly'))])

param_grid = {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__degree': [2, 3, 4]
}

grid_poly = GridSearchCV(pipe_poly, param_grid, cv=kfold)
grid_poly.fit(X_train, y_train)


pipe_logistic_reg = Pipeline([
    ('preprocessing', preprocess_pipeline),
    ('classifier', LogisticRegression(max_iter=1000))])

param_grid = {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2']
}

grid_logistic_reg = GridSearchCV(pipe_logistic_reg, param_grid, cv=kfold)
grid_logistic_reg.fit(X_train, y_train)


models = []
models.append(('SVM linear', grid_linear.best_estimator_))
models.append(('SVM rbf', grid_rbf.best_estimator_))
models.append(('SVM poly', grid_poly.best_estimator_))
models.append(('Logistic regression', grid_logistic_reg.best_estimator_))


precision_score = []
recall_score = []
f1_score = []
accuracy_score = []
for name, model in models:
    print(name)
    print("precision_score: {}".format(metrics.precision_score(y_test, model.predict(X_test)) ))
    print("recall_score: {}".format( metrics.recall_score(y_test, model.predict(X_test)) ))
    print("f1_score: {}".format( metrics.f1_score(y_test, model.predict(X_test)) ))
    print("accuracy_score: {}".format( metrics.accuracy_score(y_test, model.predict(X_test)) ))
    precision_score.append(metrics.precision_score(y_test, model.predict(X_test)))
    recall_score.append(metrics.recall_score(y_test, model.predict(X_test)))
    f1_score.append( metrics.f1_score(y_test, model.predict(X_test)))
    accuracy_score.append(metrics.accuracy_score(y_test, model.predict(X_test)))


methods = ['SVM linear', 'SVM rbf', 'SVM poly', 'Logistic regression']

d = {'Method': methods,
     'precision_score': precision_score,
     'recall_score': recall_score,
     'f1_score': f1_score,
     'accuracy_score' : accuracy_score}
df = pd.DataFrame(data=d)
print(df)