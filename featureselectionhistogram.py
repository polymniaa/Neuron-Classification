# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 08:48:57 2016

@author: sarun
"""

import pandas as pd
import numpy as np
import pylab as P
from matplotlib.ticker import FormatStrFormatter
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2 
from sklearn.feature_selection import SelectKBest

data = pd.read_csv('data.csv')
Y = np.array(data['Type'])
data = data.drop('Type', 1)
classes = np.unique(Y)

X = np.array(data, dtype='float')
X = X / X.sum(axis=0)[np.newaxis, :]

#best feature of PCA
pca = PCA(n_components=0.95)        
pca.fit(X)
maxfeat = pca.components_[0,:].argmax()
f = P.figure()
for i in range(len(classes)):
    ax = P.subplot(3,3,i+1)
    n, bins, patches = P.hist(X[Y==classes[i],maxfeat], 200, histtype ='bar')
    P.xlabel(data.columns[maxfeat])
    P.ylabel(classes[i])
    ax.locator_params(axis='x',nbins=3)
    ax.locator_params(axis='y',nbins=4)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0e'))
f.tight_layout()
P.show()

##worst feature of PCA
#pca = PCA(n_components=0.95)        
#pca.fit(X)
#minfeat = pca.components_[0,:].argmin()
#P.figure()
#for i in range(len(classes)):
#    P.subplot(3,3,i+1)
#    n, bins, patches = P.hist(X[Y==classes[i],minfeat], 200, histtype ='bar')
#    P.xlabel(data.columns[minfeat])
#    P.ylabel(classes[i])
#    P.show()
#
clf = RandomForestClassifier(n_estimators=500, max_depth=30, bootstrap=False, min_samples_split = 10, class_weight="balanced")
clf = clf.fit(X, Y)
maxfeat = clf.feature_importances_.argmax()
f = P.figure()
for i in range(len(classes)):
    ax = P.subplot(3,3,i+1)
    n, bins, patches = P.hist(X[Y==classes[i],maxfeat], 200, histtype='bar')         
    P.xlabel(data.columns[maxfeat])
    P.ylabel(classes[i])
    ax.locator_params(axis='x',nbins=3)
    ax.locator_params(axis='y',nbins=4)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0e'))
f.tight_layout()
P.show()

#
#model = SelectKBest(chi2, k=1)
#model.fit(X, Y)
#maxfeat = model.get_support(True)[0]
#f = P.figure()
#for i in range(len(classes)):
#    ax = P.subplot(3,3,i+1)    
#    n, bins, patches = P.hist(X[Y==classes[i],maxfeat], 200, histtype='bar')         
#    P.xlabel(data.columns[maxfeat])
#    P.ylabel(classes[i])
#    ax.locator_params(axis='x',nbins=3)
#    ax.locator_params(axis='y',nbins=4)
#    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0e'))
#f.tight_layout()
#P.show()
#
