import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, precision_score, recall_score

data = load_breast_cancer()
print(list(data.keys()))


X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

clf = LogisticRegression()
clf.fit(X_train, y_train)
y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3)

print(precision_score(y_train, y_train_pred))
print(recall_score(y_train, y_train_pred))
print(f1_score(y_train, y_train_pred))