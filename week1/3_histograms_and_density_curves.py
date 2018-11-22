import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd

colnames=['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width','Species']
d = pd.read_csv('iris.txt',names=colnames)

# Arrange the intervals
pdf_ticks = np.linspace(0,8,160,endpoint=False)
ticks = np.linspace(0,8,16,endpoint=False)
bins = np.linspace(0,8,16,endpoint=False)

# Get density of petal length over classes
density = gaussian_kde(d.loc[d['Species']=='setosa']['Petal.Length'])
plt.plot(pdf_ticks,density(pdf_ticks), color='r')
density = gaussian_kde(d.loc[d['Species']=='versicolor']['Petal.Length'])
plt.plot(pdf_ticks,density(pdf_ticks), color='g')
density = gaussian_kde(d.loc[d['Species']=='virginica']['Petal.Length'])
plt.plot(pdf_ticks,density(pdf_ticks), color='b')

# Draw the histogram with densities
plt.hist(d['Petal.Length'], bins=bins, density=True)
plt.xticks(ticks)
plt.ylabel('Density')
plt.ylim(0,1)
plt.xlim(0,8)
plt.show()