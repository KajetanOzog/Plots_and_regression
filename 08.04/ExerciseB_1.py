import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import datasets, linear_model


np.random.seed(0)
n_samples = 30
true_fun = lambda X: np.cos(1.5 * np.pi * X)
X = np.sort(np.random.rand(n_samples))
y = true_fun(X) + np.random.randn(n_samples) * 0.1
plt.figure(figsize=(6,6))
axes = plt.gca()
axes.set_xlim([0,1])
axes.set_ylim([-1.5,1.5])
plt.scatter(X, y,  color='black')
x_tr = np.linspace(0, 1, 200)
plt.show()
s = np.random.random_sample(n_samples)
s[s > 0.5] = 1
s[s <= 0.5] = 0

X1 = X[s == 1]
y1 = y[s == 1]
X2 = X[s == 0]
y2 = y[s == 0 ]

# Plot outputs
plt.figure(figsize=(6,6))
axes = plt.gca()
axes.set_xlim([0,1])
axes.set_ylim([-1.5,1.5])
plt.scatter(X1, y1,  color='blue')
plt.scatter(X2, y2,  color='red')
x_tr = np.linspace(0, 1, 200)
plt.show()

X1 = np.vstack(X1)
X2 = np.vstack(X2)


model1 = linear_model.LinearRegression()
model1.fit(X1, y1)
plt.figure(figsize=(6,6))
axes = plt.gca()
axes.set_ylim([-1.5, 1.5])
axes.set_xlim([0, 1])
plt.scatter(X1, y1,  color='blue')
x_plot = np.vstack(np.linspace(0, 1, 1000))
plt.plot(x_plot, model1.predict(x_plot), color='blue',linewidth=1)
plt.show()


model2 = linear_model.LinearRegression()
model2.fit(X2, y2)
plt.figure(figsize=(6,6))
axes = plt.gca()
axes.set_ylim([-1.5, 1.5])
axes.set_xlim([0, 1])
plt.scatter(X2, y2,  color='red')
x_plot = np.vstack(np.linspace(0, 1, 1000))
plt.plot(x_plot, model2.predict(x_plot), color='red',linewidth=1)
plt.show()

model20 = make_pipeline(PolynomialFeatures(20), linear_model.LinearRegression())
model20.fit(X1, y1)
plt.figure(figsize=(6, 6))
axes = plt.gca()
axes.set_ylim([-1.5, 1.5])
axes.set_xlim([0, 1])
plt.scatter(X1, y1,  color='blue')
x_plot = np.vstack(np.linspace(0, 1, 1000))
plt.plot(x_plot, model20.predict(x_plot), color='blue',linewidth=1)
plt.show()


model20_2 = make_pipeline(PolynomialFeatures(20), linear_model.LinearRegression())
model20_2.fit(X2, y2)
plt.figure(figsize=(6, 6))
axes = plt.gca()
axes.set_ylim([-1.5, 1.5])
axes.set_xlim([0, 1])
plt.scatter(X2, y2,  color='red')
x_plot = np.vstack(np.linspace(0, 1, 1000))
plt.plot(x_plot, model20_2.predict(x_plot), color='red',linewidth=1)
plt.show()
