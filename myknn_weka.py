# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:04:43 2016

@author: sarun
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
 
data = pd.read_csv('data.csv')
Y = np.array(data['Type'])
data = data.drop('Type', 1)
X = np.array(data, dtype='float')

import weka.core.jvm as jvm
jvm.start()

from weka.core.converters import Loader
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file("data.arff")
data.class_is_last()

from weka.core.dataset import Instances
import weka.core.converters
#from weka.classifiers import Classifier
#cls = Classifier(classname="weka.classifiers.laze.IBk")
#cls.options = ["-K", "10", "-W", "0"]
#print(cls.to_help())
#
#for index, inst in enumerate(data):
#    pred = cls.classify_instance(inst)
#    dist = cls.distribution_for_instance(inst)
#    print(str(index+1) + ": label index=" + str(pred) + ", class distribution=" + str(dist))

from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
#classifier = Classifier(classname="weka.classifiers.lazy.IBk", options=["-K", "10", "-W", "0", "-I"])
classifier = Classifier(classname="weka.classifiers.lazy.IBk")
classifier.options = ["-K", "10", "-W", "0", "-I", "-A", "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.ManhattanDistance -R first-last\""]
evaluation = Evaluation(data)                     # initialize with priors
evaluation.crossvalidate_model(classifier, data, 10, Random(42))  # 10-fold CV
print(evaluation.summary())
print("pctCorrect: " + str(evaluation.percent_correct))
print("incorrect: " + str(evaluation.incorrect))

#cls = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.3"])
#cls.build_classifier(data)
#preds = np.zeros()
#for index, inst in enumerate(data):
#    pred = cls.classify_instance(inst)
#    dist = cls.distribution_for_instance(inst)
#    print(str(index+1) + ": label index=" + str(pred) + ", class distribution=" + str(dist))

jvm.stop()