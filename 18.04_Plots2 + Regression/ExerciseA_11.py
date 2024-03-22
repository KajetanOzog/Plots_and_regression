import numpy as np
from scipy import optimize


def c(sigma, r):
    return np.sqrt(2/np.pi) * (sigma**-1) * ((1+r)**-1)


def SN(x, m, sigma, r):
    a = (x - m) ** 2
    b1 = 2 * (sigma ** 2)
    b2 = 2 * (sigma * r) ** 2
    if x <= m:
        return c(sigma, r) * np.exp(-(1 / b1) * a)
    else:
        return c(sigma, r) * np.exp(-(1 / b2) * a)


def log_likelihood(params, vect):
    m, sigma, r = params
    s = 0
    for x in vect:
        s += np.log(SN(x, m, sigma, r))
    return s


data = np.linspace(-4, 4, 100)
m0 = 0
s0 = 1
r0 = 0.5
x0 = np.asarray((m0, s0, r0))
res1 = optimize.fmin_cg(log_likelihood, x0, args=(data, ))
print(res1)