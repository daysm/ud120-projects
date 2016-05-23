#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

# Number of features
print "Number of features: ", len(features_train[0])
# Percentile 10: 3786 (more complex)
# Percentile 1: 379

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)

from sklearn.metrics import accuracy_score

print accuracy_score(labels_test, predictions)  # 0.977815699659


#########################################################


