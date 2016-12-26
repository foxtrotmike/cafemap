# -*- coding: utf-8 -*-
"""
Created on Thu Apr 07 22:58:15 2016

@author: afsar
"""



import numpy as np
from cafeMap import *
import matplotlib.pyplot as plt
import matplotlib as mpl


if __name__=='__main__':
    data = 'data\\Arcene\\arcene'
    X = np.loadtxt(data+'_train.data')
    Y = np.loadtxt(data+'_train.labels')
    Xv = np.loadtxt(data+'_valid.data')
    Yv = np.loadtxt(data+'_valid.labels')
    Xtemp = np.vstack((X,Xv))
    
    Ytemp = np.append(Y,Yv)
    M = np.mean(Xtemp,axis=0)
    S = np.std(Xtemp,axis=0)+1e-7
    N,d = X.shape

    X = (X.T/np.linalg.norm(X,axis=1)).T
    
    Xv = (Xv.T/np.linalg.norm(Xv,axis=1)).T    
    
    I = createInstances(X, Y)
    Iv = createInstances(Xv, Yv)
    llc = compute_gammas(I+Iv, K=50,  gamma=1e-3)
    classifier = cafeMap(Lambda = 1e-2, T = 20e3,  no_bias = False, encoder = None, c_arg=True)
    classifier.train(I, history = 500)
    scores = np.array(classifier.test(Iv))
    aidx = np.argsort(scores)    
    pidx = Yv==1
    nidx = Yv!=1
    scores = scores[aidx]
    Yv = Yv[aidx]    
    amax = 0
    for s in scores:
        a = np.mean((2*(scores[pidx]>s)-1)==Yv[pidx])
        a += np.mean((2*(scores[nidx]>s)-1)==Yv[nidx])
        a /= 2
        if a>amax:
            amax = a
    print "Max Balanced Accuracy",amax
    fpr,tpr,auc = roc(list(scores),list(Yv)) 
    

    plt.figure()
    plt.plot(fpr,tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.axis([0,1,0,1])
    plt.grid()
    plt.title(str(auc))    


    plotConvergence(classifier)
    
    