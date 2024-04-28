import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn.linear_model as lm
import matplotlib.pyplot as plt


f = lambda x: ((x*2-1)*(x**2-2)*(x-2)+3)
x_tr = np.linspace(0, 3, 200)
y_tr = f(x_tr)

x = stats.uniform(0,3).rvs(100)
y = f(x) + stats.norm(0,0.2).rvs(len(x))

M3 = np.vstack((np.ones_like(x), x, x**2, x**3)).T
p3 = np.linalg.lstsq(M3, y, rcond=None)

f_lr_3 = lambda x: p3[0][3]*pow(x, 3) + p3[0][2]*pow(x, 2) + p3[0][1] * x + p3[0][0]
x_f_lr3 = np.linspace(0., 3, 200)
y_f_lr3 = f_lr_3(x_tr)
plt.figure(figsize=(6,6))
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,8])
plt.plot(x_f_lr3, y_f_lr3, 'g')
plt.plot(x, y, 'ok', ms=10)
plt.show()

M4 = np.vstack((np.ones_like(x), x, x**2, x**3, x**4)).T
p4 = np.linalg.lstsq(M4, y, rcond=None)


f_lr_4 = lambda x: p4[0][4]*pow(x, 4) + p4[0][3]*pow(x, 3) + p4[0][2]*pow(x, 2) + p4[0][1] * x + p4[0][0]
x_f_lr4 = np.linspace(0., 3, 200)
y_f_lr4 = f_lr_4(x_tr)
plt.figure(figsize=(6,6))
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,8])
plt.plot(x_f_lr4, y_f_lr4, 'g')
plt.plot(x, y, 'ok', ms=10)
plt.show()


M5 = np.vstack((np.ones_like(x), x, x**2, x**3, x**4, x**5)).T
p5 = np.linalg.lstsq(M5, y, rcond=None)


f_lr_5 = lambda x: p5[0][5]*pow(x, 5) + p5[0][4]*pow(x, 4) + p5[0][3]*pow(x, 3) + p5[0][2]*pow(x, 2) + p5[0][1] * x + p5[0][0]
x_f_lr5 = np.linspace(0., 3, 200)
y_f_lr5 = f_lr_5(x_tr)
plt.figure(figsize=(6,6))
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,8])
plt.plot(x_f_lr5, y_f_lr5, 'g')
plt.plot(x, y, 'ok', ms=10)
plt.show()

plt.plot(x_tr[:200], y_tr[:200], '--k')
plt.plot(x_f_lr3, y_f_lr3, color='b', label="x**3")
plt.plot(x_f_lr4, y_f_lr4, color='r', label="x**4")
plt.plot(x_f_lr5, y_f_lr5, color='yellow',  label="x**5")
plt.legend()
plt.show()
