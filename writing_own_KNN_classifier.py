#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 12:00:35 2017

@author: chandru
"""

#Writing our own classifier   

#Distance between 2 points using euclidiean distance  
from scipy.spatial import distance  

def euc(a,b):
    return distance.euclidean(a,b)

#Defining Class ScrppyKNN
class ScrappyKNN():
    
#Fit method used to memorise training sets
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

#predict() inputs testing data X_test and returns list of predictions
#KNN Algorithm -> For each testing variable we will find the nearest neighbour of training set
    def predict(self,X_test):
        predictions=[]
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
        
    def closest(self,row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.Y_train[best_index]
                
    

#import a dataset
from sklearn import datasets
iris = datasets.load_iris()

#Assign input data to X and output target to Y
X = iris.data
Y = iris.target

#Partition the dataset wrt to training and testing set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

#from sklearn import tree
#my_classifier = tree.DecisionTreeClassifier()

#from sklearn.neighbors import KNeighborsClassifier
#my_classifier = KNeighborsClassifier()

#calling our own classifier
my_classifier = ScrappyKNN()

#Fiting the training data into our classifier
my_classifier.fit(X_train,Y_train)

#Predicting the testing data output based on model created wrt training dataset  
predictions = my_classifier.predict(X_test)
print(predictions)

#Evalutating accuracy score of testing set results 
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, predictions))