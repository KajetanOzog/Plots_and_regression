from matplotlib import pyplot as plt
from sklearn import datasets
import numpy as np
from scipy import optimize


def Gpdf(x, mu, sigma):
    return 1/(sigma * (2*np.pi)**.5) * np.e ** (-(x-mu)**2/(2 * sigma**2))


def G_density(vect, mu, sigma):
    density = np.array([])
    for x in vect:
        density = np.append(density, Gpdf(x, mu, sigma))
    return density


def Glog_likelihood(params, uni):
    mu, sigma = params
    s = 0
    for x in uni:
        g = Gpdf(x, mu, sigma)
        s += np.log(g)
    return s


def c(sigma, r):
    return np.sqrt(2/np.pi) * (sigma**-1) * ((1+r)**-1)


def SN_density(vect, m, sigma, r):
    density = np.array([])
    for x in vect:
        a = (x-m)**2
        b1 = 2*(sigma**2)
        b2 = 2*(sigma*r)**2

        if x <= m:
            density = np.append(density, c(sigma, r) * np.exp(-(1/b1)*a))
        else:
            density = np.append(density, c(sigma, r) * np.exp(-(1/b2)*a))
    return density


def SN(x, m, sigma, r):
    a = (x - m) ** 2
    b1 = 2 * (sigma ** 2)
    b2 = 2 * (sigma * r) ** 2
    if x <= m:
        return c(sigma, r) * np.exp(-(1 / b1) * a)
    else:
        return c(sigma, r) * np.exp(-(1 / b2) * a)


def SNlog_likelihood(params, vect):
    m, sigma, r = params
    s = 0
    for x in vect:
        s += np.log(SN(x, m, sigma, r))
    return s


california_housing = datasets.fetch_california_housing()
data = california_housing.data[:, 7]
mu = np.mean(data)
sigma = np.std(data)
tau = np.std(data)

x = np.linspace(-5, 5, data.shape[0])

x0 = np.asarray((mu, sigma))
g = optimize.fmin_cg(Glog_likelihood, x0, args=(data, ))
x0 = np.asarray((mu, sigma, tau))
sn = optimize.fmin_cg(SNlog_likelihood, x0, args=(data, ))
print(g)
print(sn)
plt.plot(x, data, color='red')
plt.plot(x, G_density(x, g[0], g[1]), color='green')
plt.plot(x, SN_density(x, sn[0], sn[1], sn[2]), color='blue')
plt.show()
plt.plot(x, G_density(x, g[0], g[1]), color='green')
plt.show()