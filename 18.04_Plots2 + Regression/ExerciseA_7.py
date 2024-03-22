import numpy as np
from scipy import optimize


def Gpdf(x, mu, sigma):
    return 1/(sigma * (2*np.pi)**.5) * np.e ** (-(x-mu)**2/(2 * sigma**2))


def log_likelihood(params, uni):
    mu, sigma = params
    s = 0
    for x in uni:
        g = Gpdf(x, mu, sigma)
        s += np.log(g)
    return s


uniform = np.random.uniform(-1, 1, 100)
m0 = 0
s0 = 1
x0 = np.asarray((m0, s0))
res1 = optimize.fmin_cg(log_likelihood, x0, args=(uniform, ))
print(res1)
