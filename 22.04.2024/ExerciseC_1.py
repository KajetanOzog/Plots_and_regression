import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


data = load_breast_cancer()
print(list(data.keys()))
X, y = data.data, data.target

plt.figure(figsize=(7, 7))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X, y)

#1
print(log_reg.coef_)
print(log_reg.intercept_)

#2
print(log_reg.predict(X))

#3
print(log_reg.predict_proba(X))

#4
print(accuracy_score(log_reg.predict(X), y))


#second part
X_train, X_test , y_train, y_test = train_test_split(X,y,test_size=0.20)

plt.figure(figsize=(7, 7))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.show()
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

#1
print(log_reg.coef_)
print(log_reg.intercept_)

#2
print(log_reg.predict(X_test))

#3
print(log_reg.predict_proba(X_test))

#4
print(accuracy_score(log_reg.predict(X_test), y_test))

