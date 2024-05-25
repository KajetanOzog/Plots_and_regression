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
plt.imshow(X_train[0], cmap="binary")
plt.axis('off')
plt.show()

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])


histories = list()
#SGD
history_sgd = History()
sgd = keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="sparse_categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, callbacks=[history_sgd])
histories.append(history_sgd)



#ADAM1
hisotory_adam1 = History()
adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss="sparse_categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, callbacks=[hisotory_adam1])
histories.append(hisotory_adam1)



#ADAM2
hisotory_adam2 = History()
adam2 = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss="sparse_categorical_crossentropy", optimizer=adam2, metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=50, callbacks=[hisotory_adam2])
histories.append(hisotory_adam2)


optimizers = ['SGD', 'Adam1', 'Adam2']
compare_val_loss(histories, optimizers)
compare_val_accuracy(histories, optimizers)