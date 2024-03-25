import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

a, b = -1, 1
n = 10000

X = np.random.uniform(a, b, n)

params = stats.norm.fit(X)

fitted_distribution = stats.norm(*params)

plt.plot(np.linspace(-5, 5, n), fitted_distribution.pdf(np.linspace(-5, 5, n)))
plt.show()