import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

colnames=['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width','Species']
d = pd.read_csv('iris.txt',names=colnames)

# Bin and tick initialization
bins = np.linspace(0,8,16,endpoint=False)
ticks = np.linspace(0,8,16,endpoint=False)


# To show the outliers in setosa's sepal width feature
#bins = np.linspace(0,8,60,endpoint=False)
#plt.hist(d.loc[d['Species']=='setosa']['Sepal.Width'], bins=bins)

# Draw the histogram of Petal Length
plt.hist(d['Petal.Length'], bins=bins)
plt.xticks(ticks)
plt.ylabel('Frekans')
plt.xlabel("Petal Length")
plt.xlim(0,8)
plt.title("Petal Length Histogram")
plt.show()