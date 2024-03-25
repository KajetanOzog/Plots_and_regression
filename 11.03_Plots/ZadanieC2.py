import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Parametry rozkładu normalnego
mean = [0, 0]  # Średnie dla obu wymiarów
covariance = [[1, 0.5], [0.5, 1]]  # Macierz kowariancji

# Tworzenie siatki punktów do obliczenia gęstości
x, y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y

# Obliczanie gęstości w punktach siatki
density = multivariate_normal(mean, covariance).pdf(pos)

# Tworzenie wykresu
plt.figure(figsize=(8, 6))
plt.contour(x, y, density, levels=10)  # Rysowanie konturów gęstości
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Kontury gęstości dwuwymiarowego rozkładu normalnego')
plt.grid(True)
plt.show()