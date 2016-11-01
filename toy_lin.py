# -*- coding: utf-8 -*-
"""
Created on Sun Apr 03 21:57:36 2016

@author: afsar
"""


import numpy as np
from cafeMap import *
import matplotlib.pyplot as plt
import matplotlib as mpl


def plotSurf(X,Y,Z):
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.3)
    cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
    cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
    cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)
    
    ax.set_xlabel('X')
    ax.set_xlim(-2, 2)
    ax.set_ylabel('Y')
    ax.set_ylim(-2, 2)
    ax.set_zlabel('Z')
    ax.set_zlim(-10, 2)
    
    plt.show()


if __name__=='__main__':
    
    d = 50
    N = 100
    nu = 10e-1
    x = np.linspace(0,1,d)
    Xp = np.repeat(np.atleast_2d(x),N,axis = 0)    
    Xn = np.repeat(np.atleast_2d(x[::-1]),N,axis = 0)    
    
    from circle import getCircle
    
    X = np.vstack((Xp,Xn))
    d = X.shape[1]
    Nu = nu*(2*np.random.rand(2*N,d)-1)
    print "NSR",np.mean(100*np.linalg.norm(Nu,axis=1)/np.linalg.norm(X,axis=1))       
    X+=Nu    
    Y = np.array([1]*N+[-1]*N)
    
    
    instances=createInstances(X, Y)
    
    classifier = cafeMap(Lambda = 1e-1, T = 1e3, no_bias = False)    
    result,folds = classifier.kFoldCV(instances, K = 5, gamma = 1e-3, folds = 5, shuffle = True, history = 100) #10-fold CV,, parallel = 3,
    scores,labels,classifiers = zip(*result)    
    Wb = np.array([c.localWb(instances) for c in classifiers])
    W = np.mean(Wb,axis = 0)[:-1]
    fpr,tpr,auc = roc_VA(zip(*(scores,labels))) 
    
    plt.figure()
    plt.plot(fpr,tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.axis([0,1,0,1])
    plt.grid()
    plt.title(str(auc))    

    
    classifier = classifiers[0]
    
    minor = False
    plt.figure()        
    aspect = np.ceil(float(W.shape[1])/W.shape[0])
    plt.imshow(np.abs(W),interpolation='none',aspect = aspect,cmap=plt.cm.Purples) 
    plt.colorbar();ax = plt.gca(); 
    ax.set_yticks(range(W.shape[0]), minor=minor);ax.set_xticks(range(W.shape[1]), minor=minor);
    if minor: ax.grid(which='minor');
    plt.grid();
    plt.xlabel('Examples: $i$');plt.ylabel('Feature Weights: $|w(x_i)|$')
    plt.figure()        
    plt.imshow(W*Y,interpolation='none',aspect = aspect,cmap=plt.cm.Purples)     
    plt.colorbar();ax = plt.gca(); 
    ax.set_yticks(range(W.shape[0]), minor=minor);ax.set_xticks(range(W.shape[1]), minor=minor);
    if minor: ax.grid(which='minor');
    plt.grid();
    plt.xlabel('Examples: $i$');plt.ylabel('Feature Weights: $y_iw(x_i)$')
    plt.figure()        
    plt.imshow(X.T,interpolation='none',aspect = aspect,cmap=plt.cm.Purples)       
    plt.colorbar();ax = plt.gca(); 
    ax.set_yticks(range(W.shape[0]), minor=minor);ax.set_xticks(range(W.shape[1]), minor=minor);
    if minor: ax.grid(which='minor');
    plt.grid();
    plt.xlabel('Examples: $i$');plt.ylabel('Features: $x_i$')
    
    
    