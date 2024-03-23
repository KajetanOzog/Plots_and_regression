import numpy as np


def compute_error(params, x, y):
    s = 0
    a, b = params
    for (xi, yi) in zip(x, y):
        s += (yi - (a*xi + b))**2
    return s


x = np.array([1, 2, 4])
y = np.array([1, 2, 3])
a = 0.64285713
b = 0.50000003
print(compute_error((a, b), x, y))
