import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]


def load_titanic_data(filename):
    return pd.read_csv(r"C:\Users\kajet\OneDrive\Pulpit\PSI\29.04.06_05_2024\8\data\titanc\{}".format(filename))


train_data = load_titanic_data("train.csv")
test_data = load_titanic_data("test.csv")

print(train_data['Survived'].value_counts())

y_train = train_data['Survived']
x_train = train_data.drop(['Survived'], axis=1)

imputer = SimpleImputer(strategy="median")

num_pipeline = Pipeline([
    ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"]),),
    ("imputer", SimpleImputer(strategy="median")),
])

num_pipeline.fit_transform(x_train)


cat_pipeline = Pipeline([
    ("select_cat", DataFrameSelector(["Embarked", "Pclass", "Cabin", "Sex"])),
    ("imputer", MostFrequentImputer()),
    ("cat_encoder", OneHotEncoder(sparse_output=False, handle_unknown='ignore')),])

cat_pipeline.fit_transform(x_train)


preprocess_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])


seed = 123
kfold = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)


param_grid = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]
}

grid_1 = GridSearchCV(Pipeline([
    ('preprocessing', preprocess_pipeline),
    ('classifier', SVC(kernel='linear'))]), param_grid, cv=kfold)

grid_1.fit(x_train, y_train)

grid_2 = GridSearchCV(Pipeline([
    ('preprocessing', preprocess_pipeline),
    ('classifier', SVC(kernel='poly'))]), param_grid, cv=kfold)

grid_2.fit(x_train, y_train)

grid_3 = GridSearchCV(Pipeline([
    ('preprocessing', preprocess_pipeline),
    ('classifier', SVC(kernel='rbf'))]), param_grid, cv=kfold)

grid_3.fit(x_train, y_train)

grid_4 = GridSearchCV(Pipeline([
    ('preprocessing', preprocess_pipeline),
    ('classifier', LogisticRegression())]), param_grid, cv=kfold)

grid_4.fit(x_train, y_train)


pred1 = grid_1.predict(test_data)
pred2 = grid_2.predict(test_data)
pred3 = grid_3.predict(test_data)
pred4 = grid_4.predict(test_data)

df1 = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': pred1})
df2 = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': pred2})
df3 = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': pred3})
df4 = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': pred4})


df1.to_csv("SVMLinear")
df2.to_csv("SVMPoly")
df3.to_csv("SVMRbf")
df4.to_csv("Logistic")
