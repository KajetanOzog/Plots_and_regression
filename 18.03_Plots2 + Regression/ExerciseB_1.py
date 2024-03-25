import numpy as np
import numpy.linalg as alg
import sklearn.linear_model as lm
import matplotlib.pyplot as plt


f = lambda x: (x**2)
x = np.array([.2, .5, .8, .9, 1.3, 1.7, 2.1, 2.7])
y = f(x) + np.random.randn(len(x))
deg = 1
xx = np.vander(x, deg + 1)
w_0 = alg.inv(xx.T @ xx)@(xx.T @ y.reshape(-1, 1))
print("a = {}\n"
      "b = {}".format(w_0[0], w_0[1]))

axes = plt.gca()
axes.set_xlim([0, 3])
axes.set_ylim([0, 8])
plt.plot(x, y, 'o',  label='Points')
x = np.linspace(0, 3, 1000)
plt.plot(x, f(x),  label='y=x^2')
f2 = lambda x: w_0[0]*x + w_0[1]
plt.plot(x, f2(x), label='Regression')
plt.legend(loc='best')
plt.show()

