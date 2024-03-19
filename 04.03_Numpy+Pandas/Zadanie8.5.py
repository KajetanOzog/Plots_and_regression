import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

inFile = 'http://ww2.amstat.org/publications/jse/datasets/babyboom.dat.txt'
data = pd.read_csv(inFile, sep='[ ]+', header=None, engine='python',names= ['sex', 'Weight', 'Minutes'])
df = data[['Minutes', 'sex', 'Weight']]

print(df)
males = df[df["sex"] == 1]
females = df[df["sex"] == 2]

plt.scatter(np.arange(len(males)), males['Weight'], color='red')
plt.scatter(np.arange(len(females)), females['Weight'], color='green')
plt.show()

plt.hist(males['Weight'], color='black', bins=55)
plt.hist(females['Weight'], color="blue", bins=55)
plt.show()

sns.kdeplot(males["Minutes"])
sns.kdeplot(females["Minutes"])
plt.show()

plt.plot(stats.cumfreq(males, numbins=25)[0])
plt.plot(stats.cumfreq(females, numbins=25)[0])
plt.show()

plt.boxplot(males["Weight"], sym='*')
plt.boxplot(females["Weight"], sym='.')
plt.show()

sns.violinplot(males["Weight"])
sns.violinplot(females["Weight"])
plt.show()








