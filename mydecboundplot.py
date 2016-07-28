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
    # convert data into 2D
    pca = PCA(n_components=2) 
    X = pca.fit_transform(X)  
    X = X[ind]
    # convert labels into number
    le = preprocessing.LabelEncoder()
    le.fit(Y)
    Y = le.transform(Y)
    # generate grid
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    x_gap = (x_max - x_min) / 10
    x_min = x_min - x_gap
    x_max = x_max + x_gap
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    y_gap = (y_max - y_min) / 10
    y_min = y_min - y_gap
    y_max = y_max + y_gap
    xx, yy = np.meshgrid(np.arange(x_min, x_max, x_gap/50),
                         np.arange(y_min, y_max, y_gap/50))
    tmp = np.c_[xx.ravel(), yy.ravel()]
    # get decision image
    Z = clf.predict(np.dot(tmp, pca.components_))
    Z = le.transform(Z)
    Z = Z.reshape(xx.shape)

    # color of each cell type    
    col = [(0.00,0.45,0.74),
           (0.85,0.33,0.10),
           (0.93,0.69,0.13),
           (0.49,0.18,0.56),
           (0.47,0.67,0.19),
           (0.30,0.75,0.93),
           (0.64,0.08,0.18),
           (1.0,0.5,0.75),
           (0.0,1.0,0.75)]
    
    
    # plot
    plt.figure()
    plt.contourf(xx, yy, Z, levels=np.arange(-0.5,9.5), 
                       colors=col, alpha=0.1)
    handlers = []
    for i in range(len(le.classes_)):
        idx = Y[ind] == i
        handlers.append(plt.scatter(X[idx, 0], X[idx, 1], edgecolor='black', 
                                    linewidth='1', facecolor=col[i]))
    plt.legend(tuple(handlers),
       tuple(le.classes_.tolist()),
       scatterpoints=1,
       loc=0,
       fontsize=8)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    plt.tight_layout()
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
    #X = X[1:3000]
    #Y = Y[1:3000]
    
    sss = StratifiedKFold(Y, 10, random_state=0)
    itr = 1
    Ypred = np.zeros(Y.shape, dtype='object')
    
    train_index, test_index = next(iter(sss))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    clf = KNeighborsClassifierknn = KNeighborsClassifier(n_neighbors=5, weights= 'distance', metric='manhattan')
    clf = clf.fit(X_train, y_train)
    plot_mydecisionbound(X, Y, test_index, clf)
    
    
  