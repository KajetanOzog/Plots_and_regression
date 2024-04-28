import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


data = load_breast_cancer()
print(list(data.keys()))


X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20)
