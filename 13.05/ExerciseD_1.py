import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Activation


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
hidden1 = model.layers[1]
print()
print()
print(hidden1.name)

weights, biases = hidden1.get_weights()

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

history1 = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))


pd.DataFrame(history1.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)
y_pred = np.argmax(model.predict(X_new), axis=-1)
print(np.array(class_names)[y_pred])
plt.figure(figsize=(7.2, 2.4))

for index, image in enumerate(X_new):
    plt.subplot(1, 3, index + 1)
    plt.imshow(image, cmap="binary", interpolation="nearest")
    plt.axis('off')
    plt.title(class_names[y_test[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()




model = keras.models.Sequential()
model.add(Flatten(input_shape=[28, 28]))
model.add(Dense(300))
model.add(BatchNormalization())
model.add(Activation("sigmoid"))
model.add(Dense(100))
model.add(BatchNormalization())
model.add(Activation("sigmoid"))
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation("softmax"))
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history2 = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))
pd.DataFrame(history2.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()


history1_dict = history1.history
history2_dict = history2.history

plt.figure(figsize=(14, 10))


plt.subplot(2, 2, 1)
plt.plot(history1_dict['accuracy'], label='Model 1 Training accuracy')
plt.plot(history2_dict['accuracy'], label='Model 2 Training accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(history1_dict['val_accuracy'], label='Model 1 Validation accuracy')
plt.plot(history2_dict['val_accuracy'], label='Model 2 Validation accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(history1_dict['loss'], label='Model 1 Training loss')
plt.plot(history2_dict['loss'], label='Model 2 Training loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(history1_dict['val_loss'], label='Model 1 Validation loss')
plt.plot(history2_dict['val_loss'], label='Model 2 Validation loss')
plt.title('Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()