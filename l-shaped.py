# -*- coding: utf-8 -*-
"""
Created on Sat Jan 09 17:11:43 2016

@author: Amina Asif

Cafe Results for L-shaped dataset
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand
import itertools
from cafeMap import * 


def createInstances(data, labels):
    inst=[Instance() for _ in range(len(data))]
    for i in range(len(data)):
        inst[i].feature_vector=data[i]
        inst[i].label=labels[i]        
    return inst

            


if __name__ == '__main__':
####################TRAINING DATA######################

    N=400
    X1=np.array([(0,x) if rand()>0.5 else (x,0) for x in rand(N)])+0.1*rand(N,2);
    X1=X1/float(np.max(X1))
    l1=[1.0]*len(X1)    

    X2=points=np.random.rand(N,2)+np.array([0.2,0.2])
    X2=X2/float(np.max(X2))
    l2=[-1.0]*len(X2)
    data=np.vstack((X1,X2))
    labels=l1+l2

    
########################################################
    ############################test instances###########################
    x=np.arange(0,1.0, 0.005)
    y=np.arange(0,1.0, 0.005)
    test_data=[]
    
    
    for i in itertools.product(x,y):
        test_data+=[i]
    test_data=np.array(test_data) 
    
    
    
#    
    anchor_pos=kmeans(X1, 5)
    anchor_neg=kmeans(X2, 5)
    anchors=np.vstack((anchor_pos[0],anchor_neg[0]))
    
    test_ins=[Instance() for i in range(len(test_data))]
    for b in range(len(test_ins)):
        test_ins[b].feature_vector=test_data[b]
        
    

    instances=createInstances(data, labels)
    
    ###########################################################
    compute_gammas(instances, K=anchors, gamma=1.0)
    compute_gammas(test_ins, K=anchors,  gamma=1.0)
    #########################################################
    c=cafeMap(T=50000, beta=0.1, Lambda=0.1)
    c.train(instances, K=anchors, gamma=1.0)
    plt.figure() 
    ########################Testing ##########################
    predictions=[]
    l_w1=[]
    l_w2=[]
    for t in test_ins:
        predictions+=[c.predict_instance(t)]
        local_weight=c.W.dot(t.gammas)
        local_bias=c.bias.dot(t.gammas)
        
        
        l_w1+=[local_weight[0]]
        l_w2+=[local_weight[1]]
    l_w1=np.array(np.absolute(l_w1))

    l_w2=np.array(np.absolute(l_w2))

    l_w1=np.reshape(l_w1, [200,200]).T    
    l_w2=np.reshape(l_w2, [200,200]).T
    l_w1=np.flipud(l_w1)
    l_w2=np.flipud(l_w2)
        
    predictions=np.array(predictions)
    predictions=np.reshape(predictions, [200,200]).T
    predictions=np.flipud(predictions)
    plt.clf()
    #=============================================================================#
    plt.imshow(predictions, extent = [0,1.0,0,1.0], cmap=plt.cm.Purples)
    plt.colorbar()
    plt.scatter(X1.T[0], X1.T[1], c='g', marker='s', s=50, label='Positive Class')
    plt.scatter(X2.T[0], X2.T[1], c='r', marker='o', s=50, label='Negative Class')
    plt.scatter(anchors[:,0],anchors[:,1], c='k', marker='^', s=100, label='Anchor Points')

    plt.xlabel('$x_1$', fontsize=18)
    plt.ylabel('$x_2$', fontsize=18)
    predictions=np.flipud(predictions)
    con=plt.contour(predictions,[0],linewidths = [2],colors=('k'),extent=[0,1.0,0,1.0], label='f(x)=0')
    con.collections[0].set_label('$f(x)=0$')
    plt.title ('Prediction Scores: $f(x)$', fontsize=18)
    plt.legend(fontsize=12)
    
    plt.figure()
    
    #============================================================================#
    plt.imshow(l_w1, extent = [0,1.0,0,1.0], cmap=plt.cm.Purples)
    plt.colorbar()
    plt.scatter(X1.T[0], X1.T[1], c='g', marker='s', s=50, label='Positive Class')
    plt.scatter(X2.T[0], X2.T[1], c='r', marker='o', s=50, label='Negative Class')
    plt.scatter(anchors[:,0],anchors[:,1], c='k', marker='^', s=100, label='Anchor Points')

    plt.xlabel('$x_1$', fontsize=18)
    plt.ylabel('$x_2$', fontsize=18)
    
    con=plt.contour(predictions,[0],linewidths = [2],colors=('k'),extent=[0,1.0,0,1.0])
    con.collections[0].set_label('$f(x)=0$')
    plt.title ('Local Weight: |$w_1$|')
    plt.legend(fontsize=12)
    
    plt.figure()
    #============================================================================#
    plt.imshow(l_w2, extent = [0,1.0,0,1.0], cmap=plt.cm.Purples)
    plt.colorbar()
    plt.scatter(X1.T[0], X1.T[1], c='g', marker='s', s=50, label='Positive Class')
    plt.scatter(X2.T[0], X2.T[1], c='r', marker='o', s=50, label='Negative Class')
    plt.scatter(anchors[:,0],anchors[:,1], c='k', marker='^', s=100, label='Anchor Points')

    plt.xlabel('$x_1$', fontsize=18)
    plt.ylabel('$x_2$', fontsize=18)
    
    con=plt.contour(predictions,[0],linewidths = [2],colors=('k'),extent=[0,1.0,0,1.0])
    con.collections[0].set_label('$f(x)=0$')
    plt.title ('Local Weight: |$w_2$|')
    plt.legend(fontsize=12)
    