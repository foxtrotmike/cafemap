# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 22:28:45 2016

@author: Amina Asif
"""

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


class Instance:
    def __init__(self):
        self.feature_vector=[]
        self.gammas=[]
        self.label=None
        self.c=1.0
        
def createInstances(data, labels):
    inst=[Instance() for _ in range(len(data))]
    for i in range(len(data)):
        inst[i].feature_vector=data[i]
        inst[i].label=labels[i]
        
    return inst


if __name__ == '__main__':
    X1=np.random.uniform(low=0.0, high=0.5, size=(500,2))
    l1=[1.0]*len(X1)
    
    X2=np.array([(np.random.uniform(low=0.0, high=0.5),np.random.uniform(low=0.5, high=1.0) )for  i in range(500) ]);
    l2=[-1.0]*len(X2)
    
    X3=np.array([(np.random.uniform(low=0.5, high=1.0),np.random.uniform(low=0.0, high=0.5) )for  i in range(500) ]);
    l3=[-1.0]*len(X3)
    
    
    
    
    X4=np.array([(np.random.uniform(low=0.5, high=1.0),np.random.uniform(low=0.5, high=1.0) )for  i in range(500) ]);
    l4=[1.0]*len(X4)
    
    
    
    data=np.vstack((X1, X2, X3, X4))
    labels=l1+l2+l3+l4

    ############################test instances###########################
    x=np.arange(0,1.0, 0.01)
    y=np.arange(0,1.0, 0.01)

    test_data=[]
    
    
    for i in itertools.product(x,y,):
       
        test_data+=[i]
    test_data=np.array(test_data) 
    
    test_ins=[Instance() for i in range(len(test_data))]
    for b in range(len(test_ins)):
        test_ins[b].feature_vector=test_data[b]
        
    all_data=np.vstack((data, test_data))
    anchors=kmeans(all_data, 4)

    instances=createInstances(data, labels)    
    ###########################################################

    compute_gammas(test_ins, K=anchors[0], k=2, gamma=1.0)
    #########################################################
    c=cafeMap(T=10000, beta=0.1, Lambda=0.001)
    c.train(instances,   K=anchors[0],k=2, gamma=1.0)
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
    l_w1=np.reshape(l_w1, [100,100]).T    
    l_w2=np.reshape(l_w2, [100,100]).T
    l_w1=np.flipud(l_w1)
    l_w2=np.flipud(l_w2)
        
    predictions=np.array(predictions)
    predictions=np.reshape(predictions, [100,100]).T
    predictions=np.flipud(predictions)
    plt.clf()
    #=============================================================================#
    plt.imshow(predictions, extent = [0,1.0,0,1.0], cmap=plt.cm.Purples)

    plt.colorbar()

    
    plt.scatter(X1.T[0], X1.T[1], c='g', marker='s', label='Positive Class')
    plt.scatter(X3.T[0], X3.T[1], c='r', marker='o')
   
    plt.scatter(X2.T[0], X2.T[1], c='red', marker='o', label='Negative Class')
    plt.scatter(X4.T[0], X4.T[1], c='g', marker='s')
   

    plt.scatter(anchors[0][:,0],anchors[0][:,1], c='k', marker='^', s=100, label='Anchor Points')

    plt.xlabel('$x_1$', fontsize=18)
    plt.ylabel('$x_2$', fontsize=18)
    predictions=np.flipud(predictions)

    con=plt.contour(predictions,[0],linewidths = [2],colors=('k'),extent=[0,1.0,0,1.0], label='f(x)=0')
    con.collections[0].set_label('$f(x)=0$')
    plt.title ('Prediction Scores: $f(x)$')
    plt.legend()
    
    plt.figure()
    
    #============================================================================#
    plt.imshow(l_w1, extent = [0,1.0,0,1.0], cmap=plt.cm.Purples)
    
    plt.colorbar()
    plt.scatter(X1.T[0], X1.T[1], c='g', marker='s', label='Positive Class')
    plt.scatter(X3.T[0], X3.T[1], c='r', marker='o')
   
    plt.scatter(X2.T[0], X2.T[1], c='red', marker='o', label='Negative Class')
    plt.scatter(X4.T[0], X4.T[1], c='g', marker='s')

    plt.scatter(anchors[0][:,0],anchors[0][:,1], c='k', marker='^', s=100, label='Anchor Points')
    plt.xlabel('$x_1$', fontsize=18)
    plt.ylabel('$x_2$', fontsize=18)
    

    con=plt.contour(predictions,[0],linewidths = [2],colors=('k'),extent=[0,1.0,0,1.0])
    con.collections[0].set_label('f(x)=0')
    plt.title ('Local Weight: |$w_1$|')
    plt.legend()
    
    plt.figure()
    #============================================================================#
    plt.imshow(l_w2, extent = [0,1.0,0,1.0], cmap=plt.cm.Purples)

    plt.colorbar()
    plt.scatter(X1.T[0], X1.T[1], c='g', marker='s', label='Positive Class')
    plt.scatter(X3.T[0], X3.T[1], c='r', marker='o')
   
    plt.scatter(X2.T[0], X2.T[1], c='red', marker='o', label='Negative Class')
    plt.scatter(X4.T[0], X4.T[1], c='g', marker='s')

    plt.scatter(anchors[0][:,0],anchors[0][:,1], c='k', marker='^', s=100, label='Anchor Points')
    plt.xlabel('$x_1$', fontsize=18)
    plt.ylabel('$x_2$', fontsize=18)
    

    con=plt.contour(predictions,[0],linewidths = [2],colors=('k'),extent=[0,1.0,0,1.0])
    con.collections[0].set_label('f(x)=0')
    plt.title ('Local Weight: |$w_2$|')
    plt.legend(fontsize=12)
    