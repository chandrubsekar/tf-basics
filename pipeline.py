#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 12:00:35 2017

@author: chandru
"""

#import a dataset
from sklearn import datasets
iris = datasets.load_iris()

# Assign input data to X and Output Target to Y variable
X = iris.data
Y = iris.target

#Paritioning testing and training set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.7)

#Import and call classifier
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

#from sklearn.neighbors import KNeighborsClassifier
#my_classifier = KNeighborsClassifier()

#fitting our parameters into the model
my_classifier.fit(X_train,Y_train)
 
#Predicting output of test data set   
predictions = my_classifier.predict(X_test)
print(predictions)

#Calculating accuracy of test data results wrt model created from training set 
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, predictions))