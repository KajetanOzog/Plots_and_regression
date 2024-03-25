import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

a, b = -1, 1
n = 10000

X = np.random.uniform(a, b, n)

mu = np.sum(X)/len(X)
sigma = np.sqrt(np.sum((X - mu)**2)/len(X))
t = np.linspace(-3, 5, 10000)


plt.hist(X, bins=1, density=True, alpha=0.6, color='g', label='Histogram próbki jednostajnej')
plt.plot(t, stats.norm.pdf(t, mu, sigma), 'k-', lw=2, label='Rozkład normalny')


plt.legend()
plt.title('Porównanie próbki jednostajnej i rozkładu normalnego')
plt.show()