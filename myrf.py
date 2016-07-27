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
#from sklearn.feature_selection import chi2
#from sklearn.feature_selection import SelectKBest

data = pd.read_csv('data.csv')
Y = np.array(data['Type'])
data = data.drop('Type', 1)
X = np.array(data, dtype='float')
classes = np.unique(Y)

sss = StratifiedKFold(Y, 10, random_state=0)
itr = 1
Ypred = np.zeros(Y.shape, dtype='object')
'Classification using Random Forest'
for train_index, test_index in sss:
    print "Iter", itr, 
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
        
   # X_new = SelectKBest(chi2, k=2).fit_transform(X_train, y_train)
    
    clf = RandomForestClassifier(n_estimators=300, max_depth=30, bootstrap=False, class_weight="balanced", min_samples_split = 10, max_features = 'log2', min_samples_leaf=2)
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