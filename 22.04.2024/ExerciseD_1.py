import matplotlib
import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score


def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")


np.random.seed(42)
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='liac-arff')
X = X.to_numpy()

X = X / 255.
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
for target, image, ax in zip(y, X, axes.ravel()):
    ax.imshow(image.reshape(28, 28), cmap=plt.cm.gist_gray)
    ax.set_title(target)
plt.show()

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()


plt.figure(figsize=(9, 9))
example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]
plot_digits(example_images, images_per_row=10)
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape)
print(y_train.shape)
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
print(X_train.shape)
print(y_train_5.shape)
print(np.unique(y_train_5))


clf = LogisticRegression()
clf.fit(X_train, y_train_5)
plt.imshow(example_images[0].reshape(28, 28), cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()
print(clf.predict([example_images[0]]))
print(clf.predict([some_digit]))


print(cross_val_score(clf, X_train, y_train_5, cv=3, scoring="accuracy"))

y_train_pred = cross_val_predict(clf, X_train, y_train_5, cv=3)
print(confusion_matrix(y_train_5, y_train_pred))
y_train_perfect_predictions = y_train_5
print(confusion_matrix(y_train_5, y_train_perfect_predictions))


print(precision_score(y_train_5, y_train_pred))
print(recall_score(y_train_5, y_train_pred))

print(f1_score(y_train_5, y_train_pred))

print(classification_report(y_train_5, y_train_pred))



#Test
print(cross_val_score(clf, X_test, y_test_5, cv=3, scoring="accuracy"))
y_test_pred = cross_val_predict(clf, X_test, y_test_5, cv=3)
print(confusion_matrix(y_test_5, y_test_pred))
y_test_perfect_predictions = y_test_5
print(confusion_matrix(y_test_5, y_test_perfect_predictions))


print(precision_score(y_test_5, y_test_pred))
print(recall_score(y_test_5, y_test_pred))

print(f1_score(y_test_5, y_test_pred))

print(classification_report(y_test_5, y_test_pred))