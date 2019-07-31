# -*- coding: utf-8 -*-
"""
Created on Thu Apr 07 22:58:15 2016

@author: afsar
"""



import numpy as np
from cafemap import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
if __name__=='__main__':
    digits = load_digits(n_class=10)
    Y = digits.target
    X = digits.data
    c1 = 5
    c2 = 9
    X = X[(Y==c1)+(Y==c2)]
    Y = Y[(Y==c1)+(Y==c2)]
    Y = 2*(Y==c1)-1
    X, Xv, Y, Yv = train_test_split(X, Y, test_size=0.33, random_state=42)
#    data = 'data\\Arcene\\arcene'
#    X = np.loadtxt(data+'_train.data')
#    Y = np.loadtxt(data+'_train.labels')
#    Xv = np.loadtxt(data+'_valid.data')
#    Yv = np.loadtxt(data+'_valid.labels')
    Xtemp = np.vstack((X,Xv))
    
    Ytemp = np.append(Y,Yv)
    M = np.mean(Xtemp,axis=0)
    S = np.std(Xtemp,axis=0)+1e-7
    N,d = X.shape

    X = (X.T/np.linalg.norm(X,axis=1)).T
    
    Xv = (Xv.T/np.linalg.norm(Xv,axis=1)).T    
    
    I = createInstances(X, Y)
    Iv = createInstances(Xv, Yv)
    llc = compute_gammas(I+Iv, K=10,  gamma=1e-1)
    classifier = cafeMap(Lambda = 5e-2, T = 50e3,  no_bias = True, encoder = None, c_arg=True)
    classifier.train(I, history = 500)
    scores = np.array(classifier.test(Iv))

    fpr,tpr,auc = roc(list(scores),list(Yv)) 
    

    plt.figure()
    plt.plot(fpr,tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.axis([0,1,0,1])
    plt.grid()
    plt.title(str(auc))    


    plotConvergence(classifier)
#%%
    D = {}
    Xd = {}
    for i in Iv:
        if i.label not in D:
            D[i.label]=[]
            Xd[i.label]=[]
        else:
            f = classifier.W.dot(i.gammas)
            D[i.label].append(f)
            Xd[i.label].append(i.feature_vector)
    
    for k in D:
        f = np.mean(D[k],axis=0)
        plt.figure()
        plt.matshow(np.abs(f).reshape((8,8)))
        plt.colorbar()
        plt.title(k)
#    plt.figure()
#    plt.matshow((np.mean(D[1],axis=0)-np.mean(D[-1],axis=0)).reshape((8,8)))