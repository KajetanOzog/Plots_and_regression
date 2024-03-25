import numpy as np


def compute_error_1(params, x, y):
    s = 0
    a, b = params
    for (xi, yi) in zip(x, y):
        s += abs(yi - (a*xi + b))
    return s


x = np.array([1, 2, 4])
y = np.array([1, 2, 3])
a = 0.66405467
b = 0.3437813
print(compute_error_1((a, b), x, y))
