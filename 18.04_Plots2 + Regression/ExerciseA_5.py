import numpy as np
import matplotlib.pyplot as plt


def Gpdf(x, mu, sigma):
    return 1/(sigma * (2*np.pi)**.5) *np.e ** (-(x-mu)**2/(2 * sigma**2))


def log_likelihood(X, mu, sigma):
    s = 0
    for x in X:
        g = Gpdf(x, mu, sigma)
        s += np.log(g)
    return s


uniform = np.random.uniform(-1, 1, 100)
data = [(0, 1.1), (0, 1), (0, 2), (1, 1), (0.5, 0.2)]
plt.plot(uniform)
plt.show()
m = np.array([])
for i, d in enumerate(data):
    likelihood = log_likelihood(uniform, d[0], d[1])
    m = np.append(m, likelihood)
    print("For mean = {} and sigma = {}, likelihood is {}".format(d[0], d[1], likelihood))
    plot = np.random.normal(d[0], d[1], 100)
    x = np.linspace(-5, 5, 100)
    p = 1/(d[1] * np.sqrt(2 * np.pi)) * np.exp(-(x - d[0])**2 / (2 * d[1]**2))
    plt.plot(x, p, 'k', linewidth=2)
    plt.show()
print(np.max(m))
