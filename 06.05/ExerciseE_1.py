import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn.linear_model as lm
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pandas as pd


f = lambda x: ((x*2-1)*(x**2-2)*(x-2)+3)
x = stats.uniform(0,3).rvs(100)
y = f(x) + stats.norm(0,0.2).rvs(len(x))
x = np.vstack(x)
x_plot = np.vstack(np.linspace(0, 10, 100))

MLP = MLPRegressor(hidden_layer_sizes=(100, 50, 10), activation='tanh', max_iter=50000, batch_size=20, learning_rate_init=0.001, learning_rate="adaptive", solver='adam')
y_rbf = MLP.fit(x,y)

# Plot outputs
plt.figure(figsize=(6,6))
axes = plt.gca()
axes.set_xlim([0, 3])
axes.set_ylim([0, 8])
plt.scatter(x, y,  color='black')
plt.plot(x_plot, MLP.predict(x_plot), color='blue',linewidth=3)
plt.show()

print(metrics.r2_score(y, MLP.predict(x)))

scores = cross_val_score(MLP, x, y, cv=5, scoring='r2')
print("Wyniki krzyżowej walidacji:")
print(scores)
print("Średni wynik krzyżowej walidacji:")
print(scores.mean())

#22222222222222222222222
df_adv = pd.read_csv('adult/advertising.csv', index_col=0)
X = df_adv[['TV', 'Radio','Newspaper']]
y = df_adv['Sales']
print(df_adv.head())

MLP = MLPRegressor(hidden_layer_sizes=(100, 50, 10), activation='tanh', max_iter=50000, batch_size=20, learning_rate_init=0.001, learning_rate="adaptive", solver='adam')
MLP.fit(X, y)
scores = cross_val_score(MLP, X, y, cv=5, scoring='r2')
print(metrics.r2_score(y_true= y, y_pred= MLP.predict(X)))
print("Wyniki krzyżowej walidacji:")
print(scores)
print("Średni wynik krzyżowej walidacji:")
print(scores.mean())

