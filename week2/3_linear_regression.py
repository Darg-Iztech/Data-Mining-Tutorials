import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

import numpy as np


#########################################################################################
#    			LINEAR REGRESSSION WITH SINGLE FEATURE 				#
#########################################################################################
#read the data
colnames=['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width','Species']
d = pd.read_csv('iris.txt',names=colnames)
#d = shuffle(d)


#select features
linearRegressionInput = d['Sepal.Length'].values[:,np.newaxis]
linearRegressionTarget = d['Species']


#encode the target values
le = preprocessing.LabelEncoder()
le.fit(["setosa", "versicolor", "virginica"])
linearRegressionTarget = le.transform(linearRegressionTarget)


#divide the data into training and test
training_input = linearRegressionInput[:-20]
training_target = linearRegressionTarget[:-20]
test_input = linearRegressionInput[-20:]
test_target = linearRegressionTarget[-20:]



#create and fit the model
linReg = LinearRegression()
linReg.fit(training_input, training_target)
print("y = ax + b\na=" + str(linReg.coef_[0]) + "  b=" + str(linReg.intercept_))


#test the model and print accuracy
print("\nAccuracy of the model: " + str(linReg.score(test_input, test_target)))


#plot the curve
setosa_target = []
versicolor_target = []
virginica_target = []
setosa = []
versicolor = []
virginica = []

predictions = linReg.predict(training_input)
i = 0
for pred in predictions:
	if abs(pred-0) < abs(pred-1) and abs(pred-0) < abs(pred-2):
		setosa.append(training_input[i])
		setosa_target.append(training_target[i])

	elif abs(pred-1) < abs(pred-0) and abs(pred-1) < abs(pred-2):
		versicolor.append(training_input[i])
		versicolor_target.append(training_target[i])

	else:
		virginica.append(training_input[i])
		virginica_target.append(training_target[i])
	i+=1

plt.scatter(setosa, setosa_target, color='r')
plt.scatter(versicolor, versicolor_target, color='g')
plt.scatter(virginica, virginica_target, color='b')
#plt.plot(training_input, linReg.predict(training_input), color='k')
plt.show()








#########################################################################################
#    			LINEAR REGRESSSION WITH SINGLE FEATURE 				#
#########################################################################################

#read the data
colnames=['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width','Species']
d = pd.read_csv('iris.txt',names=colnames)
d = shuffle(d)

#select features
linearRegressionInput = d.get_values()[:,0:4]
linearRegressionTarget = d['Species']

#encode the target values
le = preprocessing.LabelEncoder()
le.fit(["setosa", "versicolor", "virginica"])
linearRegressionTarget = le.transform(linearRegressionTarget)

#create and fit the model
linReg = LinearRegression()
scores = cross_val_score(linReg, linearRegressionInput, linearRegressionTarget, cv=10)
print("Cross Validation scores:\n" + str(scores))
print("\nMean of cross validation values:" + str(scores.mean()))


