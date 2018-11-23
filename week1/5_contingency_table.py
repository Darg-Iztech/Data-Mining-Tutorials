import matplotlib.pyplot as plt
import pandas as pd


colnames=['Sepal.Length','Sepal_Width','Petal.Length','Petal.Width','Species']
d = pd.read_csv('iris.txt',names=colnames)


discriminator_feature = 'Sepal.Width'
threshold = 3

prediction = []
values = d.get_values()

for i in range(len(values)):
	if int(values[i][1]) < threshold:
		prediction.append(["versicolor",values[i][4]])
	else:
		prediction.append(["other",values[i][4]])


print(prediction)

tp = 0
fp = 0
tn = 0
fn = 0

for i in range(len(prediction)):
	if prediction[i][0] == 'other' and (prediction[i][1] != 'versicolor'):
		tn += 1
	elif prediction[i][0] == 'other' and (prediction[i][1] == 'versicolor'):
		fn += 1
	elif prediction[i][0] == 'versicolor' and (prediction[i][1] == 'versicolor'):
		tp += 1
	elif prediction[i][0] == 'versicolor' and (prediction[i][1] != 'versicolor'):
		fp += 1

print(str("\n   ")+str(1)+  "  |"  +str("  ")+str(  0))
print(str(1)+ "| " +str(tp) +str("   ")+  str(fn))
print(str(0)+ "| " +str(fp) + str("   ")+  str(tn))

jaccard_coefficient = tp/(fp+tp+fn)

print("\nJaccard Coefficient: ",jaccard_coefficient)



table = pd.crosstab(d['Sepal_Width']<3, d['Species']=='versicolor', margins=True)
print("\n",table)

jaccard_coefficient = table.values[1][1]/(table.values[0][1]+table.values[1][1]+table.values[1][0])
print("\nJaccard Coefficient: ",jaccard_coefficient)