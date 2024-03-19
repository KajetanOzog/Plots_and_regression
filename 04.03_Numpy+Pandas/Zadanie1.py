import numpy as np

# Generowanie losowej tablicy 100 x 10
points = np.random.rand(100, 10)

# Obliczanie odległości euklidesowej między każdą parą punktów w jednej linijce
distances = np.sqrt(np.sum((points[:, np.newaxis] - points)**2, axis=2))

print(distances)