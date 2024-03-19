import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy import optimize

a, b = -2, 4
n = 1000
X = uniform_data = np.random.uniform(a, b, n)

res2 = stats.fit(-3, 5, X, bounds)