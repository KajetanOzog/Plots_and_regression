import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import History
from keras.callbacks import LearningRateScheduler


def step_decay(epoch):
    initial_rate = 0.01
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_rate * np.power(drop, np.floor((1+epoch)/epochs_drop))
    return lrate

def compare_val_loss(histories, optimizers):
    plt.figure(figsize=(10, 6))

    for i, history in enumerate(histories):
        plt.plot(history.history['val_loss'], label=f"Val Loss - {optimizers[i]}")

    plt.title("Validation Loss Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def compare_val_accuracy(histories, optimizers):
    plt.figure(figsize=(10, 6))

    for i, history in enumerate(histories):
        plt.plot(history.history['val_accuracy'], label=f"Val Accuracy - {optimizers[i]}")

    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])


histories = list()
lrate = LearningRateScheduler(step_decay)
#Adam bez
history = History()
adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss="sparse_categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=30, callbacks=[history])
histories.append(history)

#Adam z 0.001
history = History()
adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss="sparse_categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=30, callbacks=[lrate, history])
histories.append(history)


#Adam z 0.0001
history = History()
adam = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss="sparse_categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=30, callbacks=[lrate, history])
histories.append(history)


optimizers = ["Adam", "Adam2", "Adam3"]
compare_val_loss(histories, optimizers)
compare_val_accuracy(histories, optimizers)