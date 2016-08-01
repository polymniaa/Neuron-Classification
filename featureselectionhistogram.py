# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 08:48:57 2016

@author: sarun
"""

import pandas as pd
import numpy as np
import pylab as P
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
normalizer =  preprocessing.Normalizer()
X = normalizer.transform(X)


#pca = PCA(n_components=0.95)        
#pca.fit(X)
#maxfeat = pca.components_[1,:].argmax()
#P.figure()
#n, bins, patches = P.hist(X[:,maxfeat], 200, histtype='bar')         #probably wrong
#P.xlabel(data.columns[maxfeat])
#P.ylabel('no. of instances')
#P.title('Best Feature of PCA')
#P.show()


#clf = RandomForestClassifier(n_estimators=500, max_depth=30, bootstrap=False, min_samples_split = 10, class_weight="balanced")
#clf = clf.fit(X, Y)
#maxfeat = clf.feature_importances_.argmax()
#P.figure()
#n, bins, patches = P.hist(X[:,maxfeat], 200, histtype='bar')         #probably wrong
#P.xlabel(data.columns[maxfeat])
#P.ylabel('no. of instances')
#P.title('Best Feature of RF')
#P.show()


model = SelectKBest(chi2, k=1)
model.fit(X, Y)
maxfeat = model.get_support(True)[0]
P.figure()
for i in range(len(classes)):
    P.subplot(3,3,i+1)    
    n, bins, patches = P.hist(X[Y==classes[i],maxfeat], 200, histtype='bar')         #probably wrong
    P.xlabel(data.columns[maxfeat])
    P.ylabel(classes[i])
    P.show()

