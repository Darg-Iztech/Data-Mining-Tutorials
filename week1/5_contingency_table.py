import matplotlib.pyplot as plt
import pandas as pd


colnames=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Species']
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
		prediction.append([target_class,values[i][4]])
	else:
		prediction.append(["other",values[i][4]])


print(prediction)

tp = 0
fp = 0
tn = 0
fn = 0

# Calculating value of contingency table cells
for i in range(len(prediction)):
	if prediction[i][0] == 'other' and (prediction[i][1] != target_class):
		tn += 1
	elif prediction[i][0] == 'other' and (prediction[i][1] == target_class):
		fn += 1
	elif prediction[i][0] == target_class and (prediction[i][1] == target_class):
		tp += 1
	elif prediction[i][0] == target_class and (prediction[i][1] != target_class):
		fp += 1


# Cool printing
print(str("\n   ")+str(1)+  "  |"  +str("  ")+str(  0))
print(str(1)+ "| " +str(tp) +str("   ")+  str(fn))
print(str(0)+ "| " +str(fp) + str("   ")+  str(tn))

jaccard_coefficient = tp/(fp+tp+fn)

print("\nJaccard Coefficient: ",jaccard_coefficient)


# An easier way to do the things above
table = pd.crosstab(d[discriminator_feature]<threshold, d['Species']==target_class, margins=True)
print("\n",table)

jaccard_coefficient = table.values[1][1]/(table.values[0][1]+table.values[1][1]+table.values[1][0])
print("\nJaccard Coefficient: ",jaccard_coefficient)