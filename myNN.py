# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 08:48:57 2016

@author: sarun
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 

from lasagne.layers import DenseLayer, InputLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, TrainSplit


data = pd.read_csv('data.csv')
Y = np.array(data['Type'])
data = data.drop('Type', 1)
X = np.array(data, dtype='float32')
classes = np.unique(Y)

class2num = {}
for i in range(len(classes)):
    class2num[classes[i]] = i

y = np.zeros(Y.shape, dtype='int16')
for i in range(len(Y)):
    y[i] = class2num[Y[i]]

sss = StratifiedKFold(Y, 10, random_state=0)
itr = 1
Ypred = np.zeros(Y.shape, dtype='int16')
'Classification using Random Forest'
for train_index, test_index in sss:
    print "Iter", itr, 
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    layer = [('input', InputLayer),
             ('dense0', DenseLayer),
             ('output', DenseLayer)]
    clf = NeuralNet(layers=layer,
                    input_shape=(None, np.size(X,1)),
                    dense0_num_units=10,
                    output_num_units=9,
                    output_nonlinearity=softmax,
                    
                    update=nesterov_momentum,
                    update_learning_rate=0.0001,
                    update_momentum=0.9,
                    
                    train_split=TrainSplit(eval_size=0.1),
                    verbose=1,
                    max_epochs=20)
    
    clf = clf.fit(X_train, y_train)
    Ypred[test_index] = clf.predict(X_test)    
    result = clf.predict(X_train)
    tr_acc = float(np.sum(y_train==result))/float(y_train.shape[0])
    
    accuracy = float(np.sum(y_test==Ypred[test_index]))/float(y_test.shape[0])
    print " => Train Accuracy = %.4f, Accuracy = %.4f" % (tr_acc, accuracy)
    itr += 1
    break
accuracy = float(np.sum(Y==Ypred))/float(Y.shape[0])
print "=== Total accuracy = ", accuracy, ' ==='
print ''
#print clf
cm = confusion_matrix(Y, classes[Ypred], labels=classes)
print cm

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