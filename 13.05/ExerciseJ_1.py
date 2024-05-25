import tensorflow as tf
from keras.src.layers import BatchNormalization, Activation, Dropout
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

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
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model.summary()
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
model.evaluate(X_test, y_test)


###########################################
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300),
    model.add(BatchNormalization()),
    model.add(Activation("relu")),
    model.add(Dropout(0.3)),
    keras.layers.Dense(100),
    model.add(BatchNormalization()),
    model.add(Activation("relu")),
    model.add(Dropout(0.3)),
    keras.layers.Dense(10),
    model.add(BatchNormalization()),
    model.add(Activation("relu")),
    model.add(Dropout(0.3))
])
model.summary()
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
history2 = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
model.evaluate(X_test, y_test)


optimizers = ["SGD1", "SGD2"]
compare_val_loss([history, history2], optimizers)
compare_val_accuracy([history, history2], optimizers)
