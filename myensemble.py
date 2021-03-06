# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 08:48:57 2016

@author: sarun
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import VotingClassifier

data = pd.read_csv('data.csv')
Y = np.array(data['Type'])
data = data.drop('Type', 1)
X = np.array(data, dtype='float')
classes = np.unique(Y)
X = X / X.sum(axis=0)[np.newaxis, :]

sss = StratifiedKFold(Y, 10, random_state=0)
itr = 1
Ypred = np.zeros(Y.shape, dtype='object')
print 'Classification using Ensemble'
for train_index, test_index in sss:
    print "Iter", itr, 
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
        
    clf1 = KNeighborsClassifierknn = KNeighborsClassifier(n_neighbors=5, weights= 'distance', metric='manhattan')
    clf2 = RandomForestClassifier(n_estimators=300, max_depth=30, bootstrap=False, class_weight="balanced", min_samples_split = 10)
    #clf3 = tree.DecisionTreeClassifier(max_depth=10, splitter='best', min_samples_split=81)
    
    clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2)], voting='soft')
    
    clf = clf.fit(X_train, y_train)
    Ypred[test_index] = clf.predict(X_test)    
    result = clf.predict(X_train)
    tr_acc = float(np.sum(y_train==result))/float(y_train.shape[0])
        
    accuracy = float(np.sum(y_test==Ypred[test_index]))/float(y_test.shape[0])
    print " => Train Accuracy = %.4f, Accuracy = %.4f" % (tr_acc, accuracy)
    itr += 1

accuracy = float(np.sum(Y==Ypred))/float(Y.shape[0])
print "=== Total accuracy = ", accuracy, ' ==='
print ''
print clf
cm = confusion_matrix(Y, Ypred, labels=classes)
print cm
print clf

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')    