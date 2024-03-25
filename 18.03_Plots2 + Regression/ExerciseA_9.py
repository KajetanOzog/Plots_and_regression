import matplotlib.pyplot as plt
import numpy as np


def c(sigma, r):
    return np.sqrt(2/np.pi) * (sigma**-1) * ((1+r)**-1)


def SN(vect, m, sigma, r):
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


args = [(0, 1, 1), (0, 1, 0.5), (1, 0.5, 1)]
data = np.linspace(-4, 4, 100)
for a in args:
    plt.plot(data, SN(data, a[0], a[1], a[2]))
    plt.title(label="For: {}, {}, {}".format(a[0], a[1], a[2]))
    plt.show()