import numpy as np

A = np.random.multivariate_normal(np.zeros(5), np.eye(5), size=100)
print((A - np.mean(A[: np.newaxis], axis=0)) / np.std(A[:, np.newaxis], axis=0))

''' 
B = np.arange(8).reshape(4, 2)
B[3, 1] = B[3, 1] ** 2
print(B)
print(np.mean(B[:, np.newaxis], axis=0))
print()
print(np.std(B[:, np.newaxis], axis=0))
print()
print((B - np.mean(B[:, np.newaxis], axis=0)))
print()
print((B - np.mean(B[:, np.newaxis], axis=0))/np.std(B[:, np.newaxis], axis=0))
'''
