import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import  metrics
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import History
from sklearn.preprocessing import StandardScaler


print(tf.__version__)

train_set = pd.read_csv('adult/adult.data', sep=", ", header=None)
test_set = pd.read_csv('adult/adult.test', sep=", ", skiprows=1, header=None)

col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
              'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
              'wage_class']
train_set.columns = col_labels
test_set.columns = col_labels

train = train_set.replace('?', np.nan).dropna()
test = test_set.replace('?', np.nan).dropna()

train_set.head()

dataset = pd.concat([train,test])

dataset['wage_class'] = dataset.wage_class.replace({'<=50K.': 0,'<=50K':0, '>50K.':1, '>50K':1})

dataset.drop(["fnlwgt"], axis=1, inplace=True)

dataset.drop(["education"], axis=1, inplace=True)

x = dataset.groupby('native_country')["wage_class"].mean()

d = dict(pd.cut(x[x.index != " United-States"],5,labels=range(5)))

dataset['native_country'] = dataset['native_country'].replace(d)

dataset = pd.get_dummies(dataset, drop_first=True)

train = dataset.iloc[:train.shape[0]]
test = dataset.iloc[train.shape[0]:]

X_train = train.drop("wage_class", axis=1)
y_train = train.wage_class

X_test = test.drop("wage_class", axis=1)
y_test = test.wage_class


history = History()
model = Sequential()
model.add(Dense(100,activation="sigmoid", input_shape=(X_train.shape[1],)))
model.add(Dense(50,activation="sigmoid"))
model.add(Dense(10,activation="sigmoid"))
model.add(Dense(1,activation="sigmoid"))
print(model.summary())
model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])

X_train = X_train.astype('float32')
y_train = y_train.astype('float32')
X_test = X_test.astype('float32')
y_test = y_test.astype('float32')



y_train = np.expand_dims(y_train, axis=-1)
y_test = np.expand_dims(y_test, axis=-1)

#nonscaled
history = model.fit(X_train, y_train, validation_data= (X_test, y_test), batch_size=32, epochs=100)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

print(model.evaluate(X_test,y_test))
y_pred = model.predict(X_test)

y_pred = (y_pred > 0.5).astype(int)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)


#scaled
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=100)
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

print(model.evaluate(X_test, y_test))
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)
