import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import scipy.stats as stats
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import model_selection


true_fun = lambda X: np.cos(1.5 * np.pi * X)
n_samples = 20
x = np.sort(np.random.rand(n_samples))
y = true_fun(x) + np.random.randn(n_samples) * 0.1
x = np.vstack(x)
x_plot = np.vstack(np.linspace(-3, 3, 1000))

predicts = []
models = []


m = make_pipeline(PolynomialFeatures(20), linear_model.LinearRegression())
m.fit(x, y)
models.append(m)
predicts.append(m.predict(x_plot))

alphas = {1, 10000, 0.0001}
for a in alphas:
    m = make_pipeline(PolynomialFeatures(20), Lasso(alpha=a))
    m.fit(x, y)
    models.append(m)
    predicts.append(m.predict(x_plot))

for i in range(len(models)):
    plt.plot(x_plot, predicts[i], linewidth=3)
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
plt.legend()
plt.savefig('img.png')
plt.show()


param_grid = {
    'polynomialfeatures__degree': [1, 2, 3, 4, 5, 20],  # Different polynomial degrees
    'lasso__alpha': [1, 10000, 0.0001]  # Different values of alpha
}


pipe = make_pipeline(PolynomialFeatures(), Lasso())
grid_search = GridSearchCV(pipe, param_grid, cv=10, scoring='neg_mean_absolute_error')
grid_search.fit(x, y)
print("Best parameters:", grid_search.best_params_)
print("*"*23456)




df_adv = pd.read_csv('https://raw.githubusercontent.com/przem85/bootcamp/master/statistics/Advertising.csv', index_col=0)
X = df_adv[['TV', 'radio', 'newspaper']]
y = df_adv['sales']
pipe = make_pipeline(PolynomialFeatures(), Lasso())
grid_search = GridSearchCV(pipe, param_grid, cv=10, scoring='neg_mean_absolute_error')
grid_search.fit(X, y)
print("Best parameters:", grid_search.best_params_)
