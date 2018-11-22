import matplotlib.pyplot as plt
import sys
import pandas as pd


colnames=['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width','Species']
d = pd.read_csv('iris.txt',names=colnames)
feature = 'Sepal.Length'

# Show min&max and the values at quantiles
a = d.sort_values(feature)
print(a[feature].get_values()[0]) # min
print(a[feature].get_values()[int(len(a[feature].get_values())/4)])     # %25
print(a[feature].get_values()[int(len(a[feature].get_values())/2)])     # %50
print(a[feature].get_values()[int(3*(len(a[feature].get_values())/4))]) # %75
print(a[feature].get_values()[-1]) # max

# Arrange data points and draw the boxplot
data_points = [a['Sepal.Length'].get_values(), a['Sepal.Width'].get_values(), a['Petal.Length'].get_values(), a['Petal.Width'].get_values()]
#data_points = a.get_values()[:,0:4]

plt.title("Genel Boxplot")
plt.boxplot(data_points,labels=colnames[0:4])
plt.show()


# Draw boxplot for Sepal Width over classes
b = [a.loc[a['Species']=='setosa']['Sepal.Width'], a.loc[a['Species']=='virginica']['Sepal.Width'], a.loc[a['Species']=='versicolor']['Sepal.Width']]

plt.title("Sepal Width Boxplot")
plt.ylabel('Sepal Width')
plt.boxplot(b, labels=['setosa','virginica','versicolor'])
plt.show()


# Draw boxplot for Petal Length over classes
b = [a.loc[a['Species']=='setosa']['Petal.Length'], a.loc[a['Species']=='virginica']['Petal.Length'], a.loc[a['Species']=='versicolor']['Petal.Length']]

plt.title("Petal Length Boxplot")
plt.ylabel('Petal Length')
plt.boxplot(b, labels=['setosa','virginica','versicolor'])
plt.show()



