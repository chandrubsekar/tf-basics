# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 22:19:08 2017

@author: chandru
"""

"""
Lets say hello to ML
First pgm to predict whether given fruit is Orange or Apple
Inputs : Weights in gm and shape (1.Bumpy 2.Smooth)
"""

from sklearn import tree
features = [[140, 1],[130,1],[150,2],[170,2]]
labels = ["Orange","Orange","Apple","Apple"]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels) #TRAINING ALGO INCLUDED IN CLASSIFIER OBJECT FIT -> Find patterns and data;
print(clf.predict([[150,1]]))
