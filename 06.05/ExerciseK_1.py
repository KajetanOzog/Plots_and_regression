import tensorflow as tf
from keras import Sequential
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Dense
from keras.callbacks import History
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler



train_set = pd.read_csv('adult/adult.data', sep=", ", header=None)
test_set = pd.read_csv('adult/adult.test', sep=", ", skiprows=1, header=None)

col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
              'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
              'wage_class']
train_set.columns = col_labels
test_set.columns = col_labels
train = train_set.replace('?', np.nan).dropna()
test = test_set.replace('?', np.nan).dropna()
dataset = pd.concat([train,test])
dataset['wage_class'] = dataset.wage_class.replace({'<=50K.': 0, '<=50K': 0, '>50K.': 1, '>50K': 1})
dataset.drop(["fnlwgt"], axis=1, inplace=True)
dataset.drop(["education"], axis=1, inplace=True)
x = dataset.groupby('native_country')["wage_class"].mean()
d = dict(pd.cut(x[x.index != " United-States"], 5, labels=range(5)))
dataset['native_country'] = dataset['native_country'].replace(d)
dataset = pd.get_dummies(dataset,drop_first=True)
train = dataset.iloc[:train.shape[0]]
test = dataset.iloc[train.shape[0]:]
X_train = train.drop("wage_class", axis=1)
y_train = train.wage_class
X_test = test.drop("wage_class", axis=1)
y_test = test.wage_class
history = History()
model = Sequential()

history_data = []
activation_methods = ['sigmoid', 'relu', 'elu', 'tanh', keras.layers.LeakyReLU(alpha=0.01)]
for activation in activation_methods:
    keras.backend.clear_session()
    model.add(Dense(100,activation=activation, input_shape=(X_train.shape[1],)))
    model.add(Dense(50,activation=activation))
    model.add(Dense(10,activation=activation))
    model.add(Dense(1,activation=activation))
    print(model.summary())

    if activation in ['sigmoid', 'LeakyReLU']:
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    elif activation == 'relu' or activation == 'tanh':
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=["accuracy"])
    else:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    X_test = X_test.astype('float32')
    y_test = y_test.astype('float32')

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    y_train = np.expand_dims(y_train, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32,epochs=100)
    history_data.append(history.history)
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    if activation in ['sigmoid', 'tanh', 'relu', 'elu']:
        plt.title("For {}".format(activation))
    else:
        plt.title("For LeakyReLU")
    plt.show()
    model.evaluate(X_test, y_test)
'''    y_pred = model.predict(X_test)
    print(y_pred)
    y_pred = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)'''

for h, a in zip(history_data, activation_methods):
    pd.DataFrame(h).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)

plt.show()

