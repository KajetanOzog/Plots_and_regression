from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import  metrics


train_set = pd.read_csv('adult/adult.data', sep=", ", header=None)
test_set = pd.read_csv('adult/adult.test', sep=", ", skiprows=1, header=None)

col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
              'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
              'wage_class']

train_set.columns = col_labels
test_set.columns = col_labels

train = train_set.replace('?', np.nan).dropna()
test = test_set.replace('?', np.nan).dropna()


dataset = pd.concat([train, test])
dataset['wage_class'] = dataset.wage_class.replace({'<=50K.': 0, '<=50K': 0, '>50K.': 1, '>50K': 1})

dataset.drop(["fnlwgt"], axis=1, inplace=True)
dataset.drop(["education"], axis=1, inplace=True)

x = dataset.groupby('native_country')["wage_class"].mean()

d = dict(pd.cut(x[x.index != " United-States"], 5, labels=range(5)))

dataset['native_country'] = dataset['native_country'].replace(d)

dataset = pd.get_dummies(dataset, drop_first=True)

train = dataset.iloc[:train.shape[0]]
test = dataset.iloc[train.shape[0]:]

X_train = train.drop("wage_class",axis=1)
y_train = train.wage_class

X_test = test.drop("wage_class",axis=1)
y_test = test.wage_class

model = MLPClassifier((20, 10))
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
predictions = y_pred.round()
print(metrics.accuracy_score(y_test, predictions))

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
predictions = y_pred.round()
print(metrics.accuracy_score(y_test, predictions))

