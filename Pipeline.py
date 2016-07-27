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
from sklearn.svm import LinearSVC
from  sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

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
        
#   clf = tree.DecisionTreeClassifier(max_depth=1)
#   clf = RandomForestClassifier(n_estimators=250, max_depth=None, bootstrap=False, class_weight="balanced", n_jobs=4)
#    clf = clf.fit(X_train, y_train)
#    newf = np.argsort(clf.feature_importances_)
#    
#    newf = newf[0:10]#[12, 66, 58, 34, 90, 82, 54, 94, 78, 38]
#    knn = KNeighborsClassifier(n_neighbors=10, weights= 'distance',metric='manhattan')
#    knn.fit(X_train[:,newf], y_train)
#    Ypred[test_index] = knn.predict(X_test[:,newf])
    
#    clf = Pipeline([
#        ('feature_selection', SelectFromModel(LinearSVC(penalty="l2"),
#                threshold=0.04)),
#        ('classification', RandomForestClassifier(n_estimators=250, 
#                                       max_depth=None, 
#                                       bootstrap=False, 
#                                       class_weight="balanced", 
#                                       n_jobs=4))
#        ])
    
    clf = Pipeline([
        ('feature_selection', SelectKBest(chi2, k=90)), 
        ('classification', RandomForestClassifier(n_estimators=250, 
                                       max_depth=None, 
                                       bootstrap=False, 
                                       class_weight="balanced", 
                                       n_jobs=4))
        ])
    clf.fit(X_train, y_train)
    Ypred[test_index] = clf.predict(X_test)
    
    accuracy = float(np.sum(y_test==Ypred[test_index]))/float(y_test.shape[0])
    print "Accuracy = ", accuracy
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