import numpy as np
import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from numpy.random import seed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import History

seed(123)


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

dataset['wage_class'] = dataset.wage_class.replace({'<=50K.': 0,'<=50K':0, '>50K.':1, '>50K':1})

dataset.drop(["fnlwgt"],axis=1,inplace=True)

dataset.drop(["education"],axis=1,inplace=True)

x = dataset.groupby('native_country')["wage_class"].mean()

d = dict(pd.cut(x[x.index!=" United-States"],5,labels=range(5)))

dataset['native_country'] = dataset['native_country'].replace(d)

dataset = pd.get_dummies(dataset,drop_first=True)

train = dataset.iloc[:train.shape[0]]
test = dataset.iloc[train.shape[0]:]

X_train = train.drop("wage_class",axis=1)
y_train = train.wage_class

X_test = test.drop("wage_class",axis=1)
y_test = test.wage_class

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


model = Sequential()
model.add(Dense(100,activation="sigmoid",input_shape=(X_train.shape[1],)))
model.add(Dense(50,activation="sigmoid"))
model.add(Dense(10,activation="sigmoid"))
model.add(Dense(1,activation="sigmoid"))
model.summary()

#SGD
history_sgd = History()
sgd = keras.optimizers.SGD(learning_rate =0.01, momentum=0.9, nesterov=False)
model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
model.fit(X_train, y_train, validation_data= (X_test, y_test), batch_size=32,epochs=100, callbacks=[history_sgd])
plt.plot(history_sgd.history['accuracy'], label = "tarina")
plt.plot(history_sgd.history['val_accuracy'], label = "test")
plt.legend()
plt.title("SGD, nasterov = False")
plt.show()
plt.plot(history_sgd.history['loss'], label = "tarina")
plt.plot(history_sgd.history['val_loss'], label = "test")
plt.legend()
plt.title("SGD, nasterov = False")
plt.show()

#SGD2
history_sgd2 = History()
sgd2 = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss="binary_crossentropy", optimizer=sgd2, metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32,epochs=100, callbacks=[history_sgd2])
plt.plot(history_sgd2.history['accuracy'], label = "tarina")
plt.plot(history_sgd2.history['val_accuracy'], label = "test")
plt.legend()
plt.title("SGD, nasterov = True")
plt.show()
plt.plot(history_sgd2.history['loss'], label = "tarina")
plt.plot(history_sgd2.history['val_loss'], label = "test")
plt.legend()
plt.title("SGD, nasterov = True")
plt.show()

#RMS
history_rms = History()
rms = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
model.compile(loss="binary_crossentropy", optimizer=rms, metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32,epochs=100, callbacks=[history_rms])
plt.plot(history_rms.history['accuracy'], label = "tarina")
plt.plot(history_rms.history['val_accuracy'], label = "test")
plt.legend()
plt.title("RMS")
plt.show()
plt.plot(history_rms.history['loss'], label = "tarina")
plt.plot(history_rms.history['val_loss'], label = "test")
plt.legend()
plt.title("RMS")
plt.show()

#ADA GRAD
history_adaG = History()
adaG = keras.optimizers.Adagrad(learning_rate=0.01)
model.compile(loss="binary_crossentropy", optimizer=adaG, metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32,epochs=100, callbacks=[history_adaG])
plt.plot(history_adaG.history['accuracy'], label = "tarina")
plt.plot(history_adaG.history['val_accuracy'], label = "test")
plt.legend()
plt.title("ADA GRAD")
plt.show()
plt.plot(history_adaG.history['loss'], label = "tarina")
plt.plot(history_adaG.history['val_loss'], label = "test")
plt.legend()
plt.title("ADA GRAD")
plt.show()


#ADA DELTA
history_adaD = History()
adaD = keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
model.compile(loss="binary_crossentropy", optimizer=adaD, metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32,epochs=100, callbacks=[history_adaD])
plt.plot(history_adaD.history['accuracy'], label = "tarina")
plt.plot(history_adaD.history['val_accuracy'], label = "test")
plt.legend()
plt.title("ADA DELTA")
plt.show()
plt.plot(history_adaD.history['loss'], label = "tarina")
plt.plot(history_adaD.history['val_loss'], label = "test")
plt.legend()
plt.title("ADA DELTA")
plt.show()

#ADAM 1
history_adam1 = History()
adam1 = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss="binary_crossentropy", optimizer=adam1, metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32,epochs=100, callbacks=[history_adam1])
plt.plot(history_adam1.history['accuracy'], label = "tarina")
plt.plot(history_adam1.history['val_accuracy'], label = "test")
plt.legend()
plt.title("ADAM lr = 0.001")
plt.show()
plt.plot(history_adam1.history['loss'], label = "tarina")
plt.plot(history_adam1.history['val_loss'], label = "test")
plt.legend()
plt.title("ADAM lr = 0.001")
plt.show()


#ADAM 2
history_adam2 = History()
adam2 = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss="binary_crossentropy", optimizer=adam2, metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32,epochs=100, callbacks=[history_adam2])
plt.plot(history_adam2.history['accuracy'], label = "tarina")
plt.plot(history_adam2.history['val_accuracy'], label = "test")
plt.legend()
plt.title("ADAM lr = 0.0001")
plt.show()
plt.plot(history_adam2.history['loss'], label = "tarina")
plt.plot(history_adam2.history['val_loss'], label = "test")
plt.legend()
plt.title("ADAM lr = 0.0001")
plt.show()