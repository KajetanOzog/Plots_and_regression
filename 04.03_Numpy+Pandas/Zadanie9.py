import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde

mean = 0
sigma = 1

lower_bound = mean - 3 * sigma
upper_bound = mean + 3 * sigma

x = np.random.normal(mean, sigma, size=1000)
sns.kdeplot(x, color="black", fill=True)

plt.title('Gaussian Distribution')

area_under_curve = gaussian_kde(x).integrate_box_1d(lower_bound, upper_bound)

print("Area under curve: {}".format(area_under_curve / 0.997))

print("First area: {}".format(gaussian_kde(x).integrate_box_1d(mean - sigma, mean + sigma)))
print("Second area: {}".format(gaussian_kde(x).integrate_box_1d(mean - 2*sigma, mean + 2*sigma)))
print("Third area: {}".format(gaussian_kde(x).integrate_box_1d(mean - 3*sigma, mean + 3*sigma)))
plt.show()
