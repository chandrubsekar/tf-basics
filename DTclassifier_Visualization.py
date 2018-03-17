#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 22:19:08 2017

@author: chandru
"""
"""
Iris dataset : http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html#sphx-glr-auto-examples-datasets-plot-iris-dataset-py
WIKI: https://en.wikipedia.org/wiki/Iris_flower_data_set
"""
from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

iris = load_iris()
print(iris.feature_names)
print(iris.target_names)
print(iris.data[0])
print(iris.target[0])
#for i in range(len(iris.target)):
 #   print("Example %d: label %s, features %s" % (i, iris.target[i], iris.data[i]))
    

#extract test data for verification    
test_idx=[0,50,100]

#Training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis = 0)

#test data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

#Fitting data with Decision Tree Classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

print(test_data)
print(test_target)
print(clf.predict(test_data))

#Visualization Code
import graphviz 
import pydotplus
from sklearn.externals.six import StringIO
#dot_data = tree.export_graphviz(clf, out_file=None) 
#graph = graphviz.Source(dot_data) 
#graph.render("iris") 
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
    