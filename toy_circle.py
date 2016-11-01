# -*- coding: utf-8 -*-
"""
Created on Sun Apr 03 21:57:36 2016

@author: afsar
"""


import numpy as np
from cafeMap import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools


def plotSurf(X,Y,Z):
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #X, Y, Z = axes3d.get_test_data(0.05)
    
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
    
    d = 2
    N = 100
    nu = 0e-1
    x = np.linspace(0,1,d)
    Xp = np.repeat(np.atleast_2d(x),N,axis = 0)    
    Xn = np.repeat(np.atleast_2d(x[::-1]),N,axis = 0)    
    
    from circle import getCircle
    Xp,Xn = getCircle(N)
    X = np.vstack((Xp,Xn))
    d = X.shape[1]
    Nu = nu*(2*np.random.rand(2*N,d)-1)
    print "NSR",np.mean(100*np.linalg.norm(Nu,axis=1)/np.linalg.norm(X,axis=1))       
    X+=Nu    
    Y = np.array([1]*N+[-1]*N)
    

    instances=createInstances(X, Y)
    
    classifier = cafeMap(Lambda = 1e-1, T = 5e3, no_bias = False)    
    result,folds = classifier.kFoldCV(instances, K = 5, gamma = 1e-3, folds = 5, shuffle = True, history = 100, parallel = 4) #10-fold CV,, parallel = 3,
    scores,labels,classifiers = zip(*result)    
    Wb = np.array([c.localWb(instances) for c in classifiers])
    W = np.mean(Wb,axis = 0)[:-1]
    fpr,tpr,auc = roc_VA(zip(*(scores,labels))) 
    #generate vertically averaged ROC curve
    plt.figure()
    plt.plot(fpr,tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.axis([0,1,0,1])
    plt.grid()
    plt.title(str(auc))    

    
    classifier = classifiers[0]

    nplot = 20
    ex = np.min(X,axis = 0),np.max(X,axis = 0)
    ex = [ex[0][0],ex[1][0],ex[0][1],ex[1][1]]
    x=np.linspace(ex[0],ex[1], nplot)
    y=np.linspace(ex[2],ex[3], nplot)
    Z = []
    Wp = []
    for i in itertools.product(x,y):
        inst = Instance()
        inst.feature_vector = i        
        Wp.append(classifier.localWb(inst))
        Z.append(classifier.score(inst))
    Wp = np.array(Wp)
    Z = np.array(Z)
    def plotit(Z):
        Z = np.flipud(np.reshape(Z,(nplot,nplot)).T)
        plt.plot(X[Y==1,0],X[Y==1,1],'bs')
        plt.plot(X[Y==-1,0],X[Y==-1,1],'ro')
        
        plt.imshow(Z, extent = ex,cmap=plt.cm.Purples); plt.colorbar()
        plt.contour(np.flipud(Z),[0],linewidths = [2],colors=('k'),extent=ex)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
    plt.figure()
    plotit(Z)
    plt.figure()
    plotit(Wp[:,0])
    plt.figure()
    plotit(Wp[:,1])
    plt.figure()
    plotit(Wp[:,2])