import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


np.random.seed(1)
cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

plt.hist(y_train, alpha=0.5)
plt.hist(y_test, alpha=0.5)
plt.show()



models = dict()
clf1 = SVC(probability=True)
models['svc'] = clf1
clf2 = SVC(C=1, gamma=0.00001, probability=True)
models['svc_params'] = clf2
clf3 = LogisticRegression(C=1)
models['lr'] = clf3

for _, model in models.items():
    model.fit(X_train, y_train)

precision_score = []
recall_score = []
f1_score = []
accuracy_score = []
for name, model in models.items():
    print('\n' + name + '\n')
    print("R^2: {}".format(metrics.precision_score(y_test, model.predict(X_test))))
    print("recall_score: {}".format(metrics.recall_score(y_test, model.predict(X_test))))
    print("f1_score: {}".format(metrics.f1_score(y_test, model.predict(X_test))))
    print("accuracy_score: {}".format(metrics.accuracy_score(y_test, model.predict(X_test))))
    print("roc_score: {}".format(roc_auc_score(y_test, model.predict(X_test))))

    # calculate the fpr and tpr for all thresholds of the classification
    probs = model.predict_proba(X_test)
    preds = probs[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label='%s AUC = %0.10f' % (name, roc_auc))
    plt.legend(loc='lower right')
    plt.plot([-0.1, 1.1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
plt.show()