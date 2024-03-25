import numpy as np
from scipy import optimize


def compute_error_1(params, x, y):
    s = 0
    a, b = params
    for (xi, yi) in zip(x, y):
        s += abs(yi - (a*xi + b))
    return s


x = np.array([1, 2, 4])
y = np.array([1, 2, 3])
z0 = np.asarray((0, 0))
res = optimize.fmin_cg(compute_error_1, z0, args=(x, y))
print(res)
