# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 10:05:19 2016

@author: admin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedKFold
from sklearn import preprocessing

def plot_mydecisionbound(X, Y, ind, clf, title="Decision Boundary"):
    pca = PCA(n_components=2) 
    X = pca.fit_transform(X)    
    
   
    le = preprocessing.LabelEncoder()
    le.fit(Y)
    
    x_min, x_max = X[:, 0].min() - 0.00001, X[:, 0].max() + 0.00001
    y_min, y_max = X[:, 1].min() - 0.00001, X[:, 1].max() + 0.00001
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.0002),
                         np.arange(y_min, y_max, 0.0002))
    
    plt.figure()

    tmp = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(np.dot(tmp, pca.components_))

    # Put the result into a color plot
    Z = le.transform(Z)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=le.transform(Y), cmap=plt.cm.Paired)
    plt.xlabel('')
    plt.ylabel('')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv('data.csv')
    Y = np.array(data['Type'])
    data = data.drop('Type', 1)
    X = np.array(data, dtype='float')
    classes = np.unique(Y)
    normalizer =  preprocessing.Normalizer()
    X = normalizer.transform(X)
    #X = X / X.sum(axis=0)[np.newaxis, :]
    
    sss = StratifiedKFold(Y, 10, random_state=0)
    itr = 1
    Ypred = np.zeros(Y.shape, dtype='object')
    
    train_index, test_index = next(iter(sss))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    clf = KNeighborsClassifierknn = KNeighborsClassifier(n_neighbors=5, weights= 'distance', metric='manhattan')
    clf = clf.fit(X_train, y_train)
    plot_mydecisionbound(X, Y, test_index, clf)