import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import datasets, linear_model

def Average(lst):
    return sum(lst) / len(lst)

true_fun = lambda X: np.cos(1.5 * np.pi * X)


line = list()
p_2 = list()
p_3 = list()
p_4 = list()
p_5 = list()
p_6 = list()
for i in range(100):
    print("Seed number: {}".format(i))
    np.random.seed(i)
    n_samples = 100
    X = np.sort(np.random.rand(n_samples))
    y = true_fun(X) + np.random.randn(n_samples) * 0.1
    s = np.random.random_sample(n_samples)
    g = np.sort(s)
    m = g[49]
    s[s > m] = 1
    s[s <= m] = 0
    X1 = X[s == 1]
    y1 = y[s == 1]
    X2 = X[s == 0]
    y2 = y[s == 0]
    X1 = np.vstack(X1)
    X2 = np.vstack(X2)

    print("Linear error")
    model1 = linear_model.LinearRegression()
    model1.fit(X1, y1)
    line.append(metrics.r2_score(y2, model1.predict(X2)))
    print("R^2: {}".format(metrics.r2_score(y2, model1.predict(X2))))

    print("Polynomial 2 error")
    model2 = make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())
    model2.fit(X1, y1)
    p_2.append(metrics.r2_score(y2, model2.predict(X2)))
    print("R^2: {}".format(metrics.r2_score(y2, model2.predict(X2))))

    print("Polynomial 3 error")
    model3 = make_pipeline(PolynomialFeatures(3), linear_model.LinearRegression())
    model3.fit(X1, y1)
    p_3.append(metrics.r2_score(y2, model3.predict(X2)))
    print("R^2: {}".format(metrics.r2_score(y2, model3.predict(X2))))

    print("Polynomial 4 error")
    model4 = make_pipeline(PolynomialFeatures(4), linear_model.LinearRegression())
    model4.fit(X1, y1)
    p_4.append(metrics.r2_score(y2, model4.predict(X2)))
    print("R^2: {}".format(metrics.r2_score(y2, model4.predict(X2))))

    print("Polynomial 5 error")
    model5 = make_pipeline(PolynomialFeatures(5), linear_model.LinearRegression())
    model5.fit(X1, y1)
    p_5.append(metrics.r2_score(y2, model5.predict(X2)))
    print("R^2: {}".format(metrics.r2_score(y2, model5.predict(X2))))

    print("Polynomial 6 error")
    model6 = make_pipeline(PolynomialFeatures(6), linear_model.LinearRegression())
    model6.fit(X1, y1)
    p_6.append(metrics.r2_score(y2, model6.predict(X2)))
    print("R^2: {}".format(metrics.r2_score(y2, model6.predict(X2))))


print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("Linear avg error {}".format(Average(line)))
print("Polynomial 2 avg error {}".format(Average(p_2)))
print("Polynomial 3 avg error {}".format(Average(p_3)))
print("Polynomial 4 avg error {}".format(Average(p_4)))
print("Polynomial 5 avg error {}".format(Average(p_5)))
print("Polynomial 6 avg error {}".format(Average(p_6)))
plt.scatter([1,2,3,4,5,6], [Average(line), Average(p_2), Average(p_3), Average(p_4), Average(p_5), Average(p_6)])
plt.show()