import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy import optimize
import math
a, b = 0, 1
n = 10000

X = np.random.uniform(0, 1, n)

mu = np.sum(X)/len(X)
print(mu)
sigma = np.sqrt(np.sum((X - mu)**2)/len(X))
t = np.linspace(-3, 5, 10000)

plt.plot(t, stats.norm.pdf(t, mu, sigma), 'k-', lw=2, label='Rozkład normalny ($\mu=1$, $\sigma=1$)')

plt.show()
