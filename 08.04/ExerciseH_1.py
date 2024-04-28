import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn import model_selection
from sklearn import  metrics



data_url = "http://lib.stat.cmu.edu/datasets/boston"
boston = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

boston_X = np.hstack([boston.values[::2, :], boston.values[1::2, :2]])
boston_Y = boston.values[1::2, 2]

boston_X_train = boston_X[:-50]
boston_X_test = boston_X[-50:]

boston_y_train = boston_Y[:-50]
boston_y_test = boston_Y[-50:]

X = boston_X_train
y = boston_y_train

seed = 123
kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)

grid_1 = GridSearchCV(make_pipeline(PolynomialFeatures(degree=2), ElasticNet(alpha=1, random_state=seed)),
                    param_grid={'polynomialfeatures__degree': [1, 2, 3, 4],
                    'elasticnet__alpha': [0.01, 0.1, 1, 10]},
                    cv=kfold,
                    refit=True)
grid_1.fit(X, y)
print(grid_1.best_params_)
print(grid_1.cv_results_['mean_test_score'].reshape(4, -1))
plt.matshow(grid_1.cv_results_['mean_test_score'].reshape(4, -1), vmin=0, cmap="viridis")
plt.xlabel("elasticnet__alpha")
plt.ylabel("polynomialfeatures__degree")
plt.savefig("elastic")
plt.show()

grid_2 = GridSearchCV(make_pipeline(PolynomialFeatures(degree=2), Lasso(alpha=1, random_state=seed)),
                    param_grid={'polynomialfeatures__degree': [1, 2, 3, 4],
                    'lasso__alpha': [0.01, 0.1, 1, 10]},
                    cv=kfold,
                    refit=True)

grid_2.fit(X, y)
print(grid_2.best_params_)
print(grid_2.cv_results_['mean_test_score'].reshape(4, -1))
plt.matshow(grid_2.cv_results_['mean_test_score'].reshape(4, -1), vmin=0, cmap="viridis")
plt.xlabel("lasso__alpha")
plt.ylabel("polynomialfeatures__degree")
plt.savefig("lasso")
plt.show()

grid_3 = GridSearchCV(make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=1, random_state=seed)),
                    param_grid={'polynomialfeatures__degree': [1, 2, 3, 4],
                    'ridge__alpha': [0.01, 0.1, 1, 10]},
                    cv=kfold,
                    refit=True)

grid_3.fit(X, y)
print(grid_3.best_params_)
print(grid_3.cv_results_['mean_test_score'].reshape(4, -1))
plt.matshow(grid_3.cv_results_['mean_test_score'].reshape(4, -1), vmin=0, cmap="viridis")
plt.xlabel("ridge__alpha")
plt.savefig("ridge")
plt.ylabel("polynomialfeatures__degree")
plt.show()


grid_4 = GridSearchCV(make_pipeline(PolynomialFeatures(degree=2), linear_model.LinearRegression()),
                      param_grid={'polynomialfeatures__degree': [1, 2, 3, 4]},
                      cv=kfold,
                      refit=True)

grid_4.fit(X, y)
print(grid_4.best_params_)
print(grid_4.cv_results_['mean_test_score'].reshape(4, -1))
plt.matshow(grid_4.cv_results_['mean_test_score'].reshape(4, -1), vmin=0, cmap="viridis")
plt.xlabel("linear__alpha")
plt.ylabel("polynomialfeatures__degree")
plt.savefig("linear")
plt.show()


X_test=boston_X_test
y_test=boston_y_test


models = []
models.append(('ElasticNet', grid_1.best_estimator_))
models.append(('Lasso', grid_2.best_estimator_))
models.append(('Ridge', grid_3.best_estimator_))
models.append(('LR', grid_4.best_estimator_))

r2 = []
explained_variance_score = []
median_absolute_error = []
mean_squared_error = []
mean_absolute_error = []
for name, model in models:
    print(name)
    print("R^2: {}".format(metrics.r2_score(y_test, model.predict(X_test)) ))
    print("Explained variance score: {}".format( metrics.explained_variance_score(y_test, model.predict(X_test)) ))
    print("Median absolute error: {}".format( metrics.median_absolute_error(y_test, model.predict(X_test)) ))
    print("Mean squared error: {}".format( metrics.mean_squared_error(y_test, model.predict(X_test)) ))
    print("Mean absolute errors: {}".format(metrics.mean_absolute_error(y_test, model.predict(X_test)) ))
    r2.append(metrics.r2_score(y_test, model.predict(X_test)))
    explained_variance_score.append(metrics.explained_variance_score(y_test, model.predict(X_test)))
    median_absolute_error.append( metrics.median_absolute_error(y_test, model.predict(X_test)))
    mean_squared_error.append(metrics.mean_squared_error(y_test, model.predict(X_test)))
    mean_absolute_error.append(metrics.mean_absolute_error(y_test, model.predict(X_test)))



d = {'r2': r2,
     'explained_variance_score': explained_variance_score,
     'median_absolute_error': median_absolute_error,
     'mean_squared_error' : mean_squared_error,
     'mean_absolute_error' : mean_absolute_error,
    }

df = pd.DataFrame(data=d)
df.insert(loc=0, column='Method', value=['ElasticNet', 'Lasso', 'Ridge', 'LR'])
print(df)