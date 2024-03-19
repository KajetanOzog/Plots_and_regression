import numpy as np
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)

'''print(np.min(X, axis=0))
print(X - np.min(X, axis=0))
print(np.max(X, axis=0) - np.min(X, axis=0))'''
print((X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0)))

y = np.where(y == 0, -1, y)
print(y)
