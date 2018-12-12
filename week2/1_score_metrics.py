import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics


colnames=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Species']
labels = ['versicolor','setosa','virginica']
d = pd.read_csv('iris.txt',names=colnames)

# initialization of parameters
discriminator_feature = 'Sepal_Width'
target_class = "versicolor"
threshold = 2.8

prediction = []
values = d.get_values()

# prediction
for i in range(len(values)):
	if float(values[i][colnames.index(discriminator_feature)]) < threshold: # colnames.index(discriminator_feature) gives index of feature
		prediction.append(target_class)
	else:
		prediction.append("other")
		#prediction.append(values[i][4])


print("Accuracy: ",metrics.accuracy_score(prediction,d['Species'].get_values()))
print("F1-score: ",metrics.f1_score(prediction,d['Species'].get_values(), average=None, labels=labels))
print("Jaccard similarity: ",metrics.jaccard_similarity_score(prediction, d['Species'].get_values()))
print("Precision: ",metrics.precision_score(prediction,d['Species'].get_values(), average=None, labels=labels))
print("Recall: ",metrics.recall_score(prediction,d['Species'].get_values(), average=None, labels=labels))


