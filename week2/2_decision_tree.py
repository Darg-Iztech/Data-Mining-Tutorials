import pandas as pd
from sklearn import tree


colnames=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Species']
labels = ['versicolor','setosa','virginica']
d = pd.read_csv('iris.txt',names=colnames)
#d = pd.read_csv('iris_modified.txt', names=colnames)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(d.get_values()[:,0:4], d.get_values()[:,4])

'''
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")
'''

test = [[1.5, 2, 1.5, 3],  #setosa
		[7.9, 3.8, 6.4, 2], #virginica
		[5.2 ,1, 1, 3.2],  #setosa
		[5.2, 2.7, 3.9, 1.4], #versicolor
		[5.8, 2.7, 4.1, 1], #versicolor
		[5.9, 3, 5.1, 1.8], #virginica
		[5.9, 3, 5.1, 1.8]] #virginica

print(clf.predict(test))

'''
test = [[5,3.4,1.5,0.2],  #"setosa"
[4.4,2.9,1.4,0.2],  #"setosa"
[4.9,3.1,1.5,0.1],  #"setosa"
[5.4,3.4,1.7,0.2],  #"setosa"
[5.1,3.7,1.5,0.4],  #"setosa"
[4.6,3.6,1,0.2],  #"setosa"
[5.1,3.8,1.6,0.2],  #"setosa"
[4.6,3.2,1.4,0.2],  #"setosa"
[5.3,3.7,1.5,0.2],  #"setosa"
[5,3.3,1.4,0.2],  #"setosa"
[4.9,2.4,3.3,1],  #"versicolor"
[6.6,2.9,4.6,1.3],  #"versicolor"
[5.2,2.7,3.9,1.4],  #"versicolor"
[5,2,3.5,1],  #"versicolor"
[5.6,3,4.5,1.5],  #"versicolor"
[5.8,2.7,4.1,1],  #"versicolor"
[6.2,2.2,4.5,1.5],  #"versicolor"
[6.3,2.3,4.4,1.3],  #"versicolor"
[5.6,3,4.1,1.3],  #"versicolor"
[5.5,2.5,4,1.3],  #"versicolor"
[5.9,3,5.1,1.8],  #"virginica"
[6.9,3.1,5.4,2.1],  #"virginica"
[6.4,2.8,5.6,2.1],  #"virginica"
[4.9,2.5,4.5,1.7],  #"virginica"
[7.3,2.9,6.3,1.8],  #"virginica"
[6.7,2.5,5.8,1.8],  #"virginica"
[7.2,3.6,6.1,2.5],  #"virginica"
[6,2.2,5,1.5],  #"virginica"
[6.9,3.2,5.7,2.3],  #"virginica"
[6.4,2.8,5.6,2.2]]  #"virginica"

correct_labels = ['setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa',
'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor',
'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica']
'''



