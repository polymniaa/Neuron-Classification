# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:04:43 2016

@author: sarun
"""

import os
import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from weka.classifiers import Classifier
import weka.core.jvm as jvm
from weka.core.converters import Loader


from np2arff import write_to_weka

jvm.start(max_heap_size="1024m")

data = pd.read_csv('data.csv')
Y = np.array(data['Type'])
X = np.array(data)
X[:, 0:-1] = X[:, 0:-1] - np.mean(X[:, 0:-1], axis=0)[np.newaxis, :]
X[:, 0:-1] = X[:, 0:-1] / np.var(X[:, 0:-1], axis=0)[np.newaxis, :]

classes = np.unique(Y)

sss = StratifiedKFold(Y, 10, random_state=0)
itr = 1
Ypred = np.zeros(Y.shape, dtype='object')
print "Classification using K Nearest Neighbors"
for train_index, test_index in sss:
    print "Iter", itr,
    X_train, X_test = X[train_index], X[test_index]
    X_test[:,-1] = classes[0]       # make sure test classes is removed
    y_test = Y[test_index]
    write_to_weka('train.arff', 'training_data', data.columns, X_train, classes)
    write_to_weka('test.arff', 'testing_data', data.columns, X_test, classes)

    loader = Loader(classname="weka.core.converters.ArffLoader")
    trdata = loader.load_file("train.arff")
    trdata.class_is_last()

    classifier = Classifier(classname="weka.classifiers.lazy.IBk")
    classifier.options = ["-K", "10", "-W", "0", "-I", "-A",
                          "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.ManhattanDistance -R first-last\""]
    classifier.build_classifier(trdata)

    tedata = loader.load_file("test.arff")
    tedata.class_is_last()

    for index, inst in enumerate(tedata):
        result = classifier.classify_instance(inst)
        Ypred[test_index[index]] = classes[int(result)]

    accuracy = float(np.sum(y_test == Ypred[test_index])) / float(y_test.shape[0])
    print " => Accuracy = ", accuracy
    itr += 1
accuracy = float(np.sum(Y == Ypred)) / float(Y.shape[0])
print "Total accuracy = ", accuracy

os.remove('train.arff')
os.remove('test.arff')
jvm.stop()
