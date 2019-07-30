# -*- coding: utf-8 -*-
"""
Created on Tue Apr 05 03:12:42 2016

@author: afsar
"""
from .instance import Instance
import numpy as np 
from .llc import LLC as Encoder #LocalEncoder
import matplotlib.pyplot as plt
def createInstances(data, labels, **kwargs):    
    inst=[Instance() for _ in range(len(data))]
    pos = np.sum(np.array(labels) == 1.0)
    neg = np.sum(np.array(labels) == -1.0)
    for i in range(len(data)):
        inst[i].feature_vector=data[i]
        inst[i].label=labels[i]
        if 'c'in  kwargs:
            
            inst[i].c=kwargs['c'][i]
        else:
            if inst[i].label==1.0:
                inst[i].c=1.0/(2*pos)
            else:
                inst[i].c=1.0/(2*neg)        
    return inst
def plotConvergence(cc):  
    t,v,a,d = zip(*cc.history)
    t = a
    v = np.array(v).T;
    plt.figure();
    plt.plot(t,v[0],'ro-')
    plt.xlabel('Number of Data Accesses'); plt.ylabel('Structural Risk') 
    plt.grid()
    plt.figure()
    plt.plot(t,d,'ro-')
    plt.xlabel('Number of Data Accesses'); plt.ylabel('Average number of non-zero features') 
    plt.grid()
    
    
      
     
def compute_gammas(instances, **kwargs):
    data_points=[]
    for i in instances:
        i.gammas=[]
        data_points+=[i.feature_vector]
    
        
    X=np.array(data_points)
    llc = Encoder(X,**kwargs)
    G = llc.encode(X)
    for j in range(len(instances)):
        instances[j].gammas=G[j]#instances[j].feature_vector#
    return llc
