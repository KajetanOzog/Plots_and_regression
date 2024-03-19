import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

np.random.seed(0)
sample_size = 1000
sample = np.random.uniform(0, 1, size=(sample_size, 2))

mean = np.mean(sample, axis=0)

cov = np.cov(sample, rowvar=False)

print(mean)
print(cov)

eigenvalues, eigenvectors = np.linalg.eig(cov)
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))
Z = multivariate_normal(mean, cov).pdf(pos)
plt.figure(figsize=(8, 6))
plt.scatter(Z[:, 0], Z[:, 1], alpha=0.5, label='Próbka')
plt.contour(X, Y, Z, cmap='viridis')


plt.quiver(mean[0], mean[1], eigenvectors[0, 0]*np.sqrt(eigenvalues[0]), eigenvectors[1, 0]*np.sqrt(eigenvalues[0]), angles='xy', scale_units='xy', scale=1, color='red', label='Wektor własny 1')
plt.quiver(mean[0], mean[1], eigenvectors[0, 1]*np.sqrt(eigenvalues[1]), eigenvectors[1, 1]*np.sqrt(eigenvalues[1]), angles='xy', scale_units='xy', scale=1, color='blue', label='Wektor własny 2')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Próbka, poziomice i wektory własne macierzy kowariancji')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
