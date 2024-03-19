import numpy as np

A = np.random.randint(5, 16, 100)
print(A)
print(np.bincount(A))
print(np.argmax(np.bincount(A)))

