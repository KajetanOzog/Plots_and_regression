import numpy as np
from scipy import optimize


def compute_error(params, x, y):
    s = 0
    a, b = params
    for (xi, yi) in zip(x, y):
        s += (yi - (a*xi + b))**2
    return s


x = np.array([1, 2, 4])
y = np.array([1, 2, 3])
z0 = np.asarray((1, 1))
res = optimize.fmin_cg(compute_error, z0, args=(x, y))
print(res)
