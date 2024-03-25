import numpy as np


def Gpdf(x, mu, sigma):
    return 1/(sigma * (2*np.pi)**.5) *np.e ** (-(x-mu)**2/(2 * sigma**2))


def log_likelihood(X, mu, sigma):
    s = 0
    for i in X:
        g = Gpdf(i, mu, sigma)
        s += np.log(g)
    return s


a, b = -1, 1
n = 10000

A = np.random.uniform(a, b, n)

print(log_likelihood(A, 0, 1))
