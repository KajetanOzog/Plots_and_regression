import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


def compute_error_1(params, x, y):
    s = 0
    a, b = params
    for (xi, yi) in zip(x, y):
        s += abs(yi - (a*xi + b))
    return s


def compute_error(params, x, y):
    s = 0
    a, b = params
    for (xi, yi) in zip(x, y):
        s += (yi - (a*xi + b))**2
    return s



f = lambda x: (x)
x_tr = np.linspace(0., 3, 200)
y_tr = f(x_tr)
x = np.array([.2, .5, .8, .9, 1.3, 1.7, 2.1, 2.7])
y = f(x) + np.random.randn(len(x))/5
y[1] = y[1]+10
plt.figure(figsize=(6, 6))
axes = plt.gca()
axes.set_xlim([0, 3])
axes.set_ylim([0, 12])
plt.plot(x_tr, y_tr, '--k')
plt.plot(x, y, 'ok', ms=10)

z0 = np.asarray((0, 0))
res1 = optimize.fmin_cg(compute_error_1, z0, args=(x, y))

z0 = np.asarray((0, 0))
res2 = optimize.fmin_cg(compute_error, z0, args=(x, y))

plt.plot(x, res2[0] * x + res2[1], color='r', label='compute_error')
plt.plot(x, res1[0] * x + res1[1], color='b', label='compute_error_1')

plt.show()
