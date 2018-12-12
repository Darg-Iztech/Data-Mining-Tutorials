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
linearRegressionTarget = d['Petal.Length']


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
linReg2 = LinearRegression()
scores = cross_val_score(linReg2, linearRegressionInput, linearRegressionTarget, cv=10)
print("Cross Validation scores:" + str(scores))
print("Mean of cross validation values:" + str(scores.mean()))
print("Score of the model: " + str(linReg.score(test_input, test_target)))


#plot the line
plt.scatter(training_input, training_target, color='g')
plt.plot(training_input, linReg.predict(training_input), color='k')
plt.show()








#########################################################################################
#    			LINEAR REGRESSSION WITH MULTIPLE FEATURE 				#
#########################################################################################

#read the data
colnames=['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width','Species']
d = pd.read_csv('iris.txt',names=colnames)
d = shuffle(d)

#select features
linearRegressionInput = d.get_values()[:,0:3]
linearRegressionTarget = d['Petal.Width']

#create and fit the model
linReg = LinearRegression()
scores = cross_val_score(linReg, linearRegressionInput, linearRegressionTarget, cv=10)
print("\n\nCross Validation scores:\n" + str(scores))
print("\nMean of cross validation values:" + str(scores.mean()))


