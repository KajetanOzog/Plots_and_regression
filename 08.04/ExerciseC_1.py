import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn.linear_model as lm
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import  metrics


x = stats.uniform(0,3).rvs(100)
f = lambda x: ((x*2-1)*(x**2-2)*(x-2)+3)
y = f(x) + stats.norm(0,0.3).rvs(len(x))
plt.plot(x, y, 'ok', ms=10);
plt.savefig("img.png")
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
X_train = np.vstack(X_train)
X_test = np.vstack(X_test)


model1 = linear_model.LinearRegression()
model1.fit(X_train, y_train)
print("*"*30)
print("From linear regression:")
print("Explained variance score: {}".format( metrics.explained_variance_score(y_test, model1.predict(X_test)) ))
print("Explained variance score: {}".format( metrics.explained_variance_score(y_test, model1.predict(X_test)) ))
print("Median absolute error: {}".format( metrics.median_absolute_error(y_test, model1.predict(X_test)) ))
print("Mean squared error: {}".format( metrics.mean_squared_error(y_test, model1.predict(X_test)) ))
print("Mean absolute errors: {}".format(metrics.mean_absolute_error(y_test, model1.predict(X_test)) ))
print()


model2 = make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())
model2.fit(X_train, y_train)
print("*"*20)
print("From polynomial 2 regression:")
print("Explained variance score: {}".format( metrics.explained_variance_score(y_test, model2.predict(X_test)) ))
print("Explained variance score: {}".format( metrics.explained_variance_score(y_test, model2.predict(X_test)) ))
print("Median absolute error: {}".format( metrics.median_absolute_error(y_test, model2.predict(X_test)) ))
print("Mean squared error: {}".format( metrics.mean_squared_error(y_test, model2.predict(X_test)) ))
print("Mean absolute errors: {}".format(metrics.mean_absolute_error(y_test, model2.predict(X_test)) ))
print()


model3 = make_pipeline(PolynomialFeatures(3), linear_model.LinearRegression())
model3.fit(X_train, y_train)
print("*"*20)
print("From polynomial 5 regression:")
print("Explained variance score: {}".format( metrics.explained_variance_score(y_test, model3.predict(X_test)) ))
print("Explained variance score: {}".format( metrics.explained_variance_score(y_test, model3.predict(X_test)) ))
print("Median absolute error: {}".format( metrics.median_absolute_error(y_test, model3.predict(X_test)) ))
print("Mean squared error: {}".format( metrics.mean_squared_error(y_test, model3.predict(X_test)) ))
print("Mean absolute errors: {}".format(metrics.mean_absolute_error(y_test, model3.predict(X_test)) ))
print()


model4 = make_pipeline(PolynomialFeatures(4), linear_model.LinearRegression())
model4.fit(X_train, y_train)
print("*"*20)
print("From polynomial 4 regression:")
print("Explained variance score: {}".format( metrics.explained_variance_score(y_test, model4.predict(X_test)) ))
print("Explained variance score: {}".format( metrics.explained_variance_score(y_test, model4.predict(X_test)) ))
print("Median absolute error: {}".format( metrics.median_absolute_error(y_test, model4.predict(X_test)) ))
print("Mean squared error: {}".format( metrics.mean_squared_error(y_test, model4.predict(X_test)) ))
print("Mean absolute errors: {}".format(metrics.mean_absolute_error(y_test, model4.predict(X_test)) ))
print()


model5 = make_pipeline(PolynomialFeatures(5), linear_model.LinearRegression())
model5.fit(X_train, y_train)
print("*"*20)
print("From polynomial 5 regression:")
print("Explained variance score: {}".format( metrics.explained_variance_score(y_test, model5.predict(X_test)) ))
print("Explained variance score: {}".format( metrics.explained_variance_score(y_test, model5.predict(X_test)) ))
print("Median absolute error: {}".format( metrics.median_absolute_error(y_test, model5.predict(X_test)) ))
print("Mean squared error: {}".format( metrics.mean_squared_error(y_test, model5.predict(X_test)) ))
print("Mean absolute errors: {}".format(metrics.mean_absolute_error(y_test, model5.predict(X_test)) ))
print()


model25 = make_pipeline(PolynomialFeatures(25), linear_model.LinearRegression())
model25.fit(X_train, y_train)
print("*"*20)
print("From polynomial 25 regression:")
print("Explained variance score: {}".format( metrics.explained_variance_score(y_test, model25.predict(X_test)) ))
print("Explained variance score: {}".format( metrics.explained_variance_score(y_test, model25.predict(X_test)) ))
print("Median absolute error: {}".format( metrics.median_absolute_error(y_test, model25.predict(X_test)) ))
print("Mean squared error: {}".format( metrics.mean_squared_error(y_test, model25.predict(X_test)) ))
print("Mean absolute errors: {}".format(metrics.mean_absolute_error(y_test, model25.predict(X_test)) ))
print()