import numpy as np
import matplotlib.pyplot as plt

# Parametry rozkładu normalnego
mean = [0, 0]  # Średnie dla obu wymiarów
covariance = [[1, 0.5], [0.5, 1]]  # Macierz kowariancji

# Wygenerowanie próbki z dwuwymiarowego rozkładu normalnego
sample = np.random.multivariate_normal(mean, covariance, 1000)

# Wykres próbki
plt.figure(figsize=(8, 6))
plt.scatter(sample[:, 0], sample[:, 1], s=5)  # Wykres punktowy próbki
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Losowa próbka z dwuwymiarowego rozkładu normalnego')
plt.grid(True)
plt.show()