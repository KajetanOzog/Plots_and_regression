from scipy import optimize
import numpy as np


def f(x):
    x1, x2 = x
    return (x1+1)**2+(x2)**2


x0 = np.asarray((1, 0))  # Initial guess.
res1 = optimize.fmin_cg(f, x0)
print(res1)