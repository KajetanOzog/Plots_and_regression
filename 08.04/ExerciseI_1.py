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
from sklearn.model_selection import cross_val_score

data_url = "http://lib.stat.cmu.edu/datasets/boston"
boston = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
boston_X = np.hstack([boston.values[::2, :], boston.values[1::2, :2]])
boston_Y = boston.values[1::2, 2]
X=boston_X
y=boston_Y
seed=123
kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)
grid_1 = GridSearchCV(estimator=make_pipeline(PolynomialFeatures(degree=2), ElasticNet(alpha=1, tol=0.1)),
                      param_grid={'polynomialfeatures__degree': [1, 2, 3, 4],
                      'elasticnet__alpha': [1., 2., 3.]},
                      scoring='r2',
                      cv=kfold,
                      n_jobs=-1)
scores_1 = cross_val_score(grid_1, X, y, scoring='r2', cv=5)
print('CV ElasticNet R2: %.3f +/- %.3f' % (np.mean(scores_1), np.std(scores_1)))

grid_2 = GridSearchCV(estimator=make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=1, tol=0.1)),
                      param_grid={'polynomialfeatures__degree': [1, 2, 3, 4],
                      'ridge__alpha': [1., 2., 3.]},
                      scoring='r2',
                      cv=kfold,
                      n_jobs=-1)
scores_2 = cross_val_score(grid_2, X, y, scoring='r2', cv=5)
print('CV Ridge R2: %.3f +/- %.3f' % (np.mean(scores_2), np.std(scores_2)))

grid_3 = GridSearchCV(estimator=make_pipeline(PolynomialFeatures(degree=2), Lasso(alpha=1, tol=0.1)),
                      param_grid={'polynomialfeatures__degree': [1, 2, 3, 4],
                      'lasso__alpha': [1., 2., 3.]},
                      scoring='r2',
                      cv=kfold,
                      n_jobs=-1)
scores_3 = cross_val_score(grid_3, X, y, scoring='r2', cv=5)
print('CV Lasso R2: %.3f +/- %.3f' % (np.mean(scores_3), np.std(scores_3)))

grid_4 = GridSearchCV(estimator=make_pipeline(PolynomialFeatures(degree=2), linear_model.LinearRegression()),
                      param_grid={'polynomialfeatures__degree': [1, 2, 3, 4]},
                      scoring='r2',
                      cv=kfold,
                      n_jobs=-1)

scores_4 = cross_val_score(grid_4, X, y, scoring='r2', cv=5)
print('CV Linear R2: %.3f +/- %.3f' % (np.mean(scores_4), np.std(scores_4)))

d = {'mean r2': [np.mean(scores_1), np.mean(scores_2), np.mean(scores_3), np.mean(scores_4)],
     'var r2:': [np.std(scores_1), np.std(scores_2), np.std(scores_3), np.std(scores_4)]
    }

df = pd.DataFrame(data=d)
df.insert(loc=0, column='Method', value=['ElasticNet', 'Ridge', 'Lasso', 'LR'])
print(df)
