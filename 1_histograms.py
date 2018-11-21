import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd

colnames=['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width','Species']
d = pd.read_csv('iris.txt',names=colnames)

bins = np.linspace(0,8,145,endpoint=False)
ticks = np.linspace(0,8,16,endpoint=False)

density = gaussian_kde(d.loc[d['Species']=='setosa']['Petal.Length'])
plt.plot(bins,density(bins), color='r')
density = gaussian_kde(d.loc[d['Species']=='versicolor']['Petal.Length'])
plt.plot(bins,density(bins), color='g')
density = gaussian_kde(d.loc[d['Species']=='virginica']['Petal.Length'])
plt.plot(bins,density(bins), color='b')

bins = np.linspace(0,8,16,endpoint=False)

plt.hist(d['Petal.Length'], bins=bins, density=True)
plt.xticks(ticks)
plt.ylabel('Density')

plt.ylim(0,1)
plt.show()