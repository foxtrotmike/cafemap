# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 05:43:29 2015

@author: Amina Asif / Fayyaz Minhas (updated)
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand
from .llc import *
import random
from .cv import *
from .llc import LLC 
class cafeMap:
    CV = CV 
    test = test
    kFoldCV = kFoldCV
    trainTest = trainTest
    LOOCV = LOOCV
    def __init__(self,arg = None, **kwargs):
        self.W=None
        self.bias=None
        self.Z=None
        self.Lambda = kwargs.get('Lambda',0.001)
        self.T = kwargs.get('T',10000)
        self.beta = kwargs.get('beta',0.25) #should be (||x||^2)/4
        self.no_bias = kwargs.get('no_bias',False)
        self.encoder = kwargs.get('encoder',None)
        self.history = []
        self.iters = 0
        if isinstance(arg,self.__class__):            
            self.W=arg.W
            self.Z=arg.Z
            self.bias=arg.bias
            self.Lambda = arg.Lambda
            self.T = arg.T
            self.beta = arg.beta
            self.no_bias = arg.no_bias
            self.encoder = arg.encoder
            self.history = arg.history
            self.iters = arg.iters
            
    
    def score(self, instance, **kwargs):      
       X = instance.feature_vector  
       wb = self.localWb(instance)
       score = np.dot(X,wb[:-1])+wb[-1]
       return score[0]

        
    def train(self, instances, **kwargs):
        #apply llc        
        history = kwargs.pop('history',False)
        div_by_cc=kwargs.pop('c_arg',False)
        X = np.array([i.feature_vector for i in instances])   
        
        if self.encoder is not None and not len(instances[0].gammas): 
            
            nc = kwargs.pop('K',10)
            if type(nc)!=type(0):
                K = nc
            elif nc == 0:
                K = X
            else:                    
                if nc<2:
                    nc = 2
                if nc > X.shape[0]:
                    nc = X.shape[0]
                pidx = []
                nidx = []
                for j,i in enumerate(instances):
                    if i.label == 1:
                        pidx.append(j)
                    else:
                        nidx.append(j)
                random.shuffle(pidx)
                random.shuffle(nidx)
                pidx = pidx[:min(nc/2,len(pidx))] 
                nidx = nidx[:min(nc-len(pidx),len(nidx))]
                K = X[pidx+nidx,:]                
            self.encoder = LLC(X,K = K,**kwargs)
            G = self.encoder.encode(X,**kwargs)
            for j in range(len(instances)):
                instances[j].gammas=G[j]#instances[j].feature_vector#
                
            
        #beta = self.beta/(np.max(np.linalg.norm(X,axis=1))**2)
#        MX = np.max(np.linalg.norm(X,axis=1))**2#np.max(X**2)#
#        MG = np.max(np.linalg.norm(G,axis=1))**2#np.max(G**2)#
        
        beta = self.beta
#        if MX > 1:
#            beta *= MX
#        if MG > 1:
#            beta *= MG        
        T = self.T
        self.__beta__ = beta
        
        len_g = len(instances[0].gammas)
        len_f = len(instances[0].feature_vector)
        if self.W is None: # if not then the classifier simply continues training
            self.W=np.zeros((len_f, len_g ))
            self.bias=np.zeros(len_g)
            self.Z=np.zeros(len(instances))
        else:
            print('Using exisiting weight vector with norm:',np.linalg.norm(self.W,ord = 1))
            self.Z=self.test(instances)
        
        Lambda = self.Lambda/self.W.shape[1]
#        Lambda = self.Lambda
        l_div_b=Lambda/beta
        cc = np.sum([i.c for i in instances])
        
#        import pdb; pdb.set_trace()
        for t in range(int(T)):
            if history and ((self.iters+t)%history)==0:
                Wb = self.localWb(instances)[:-1]
                fn = np.mean(np.sum(np.abs(Wb)>0,axis=0))
                self.__logHistory__((self.iters+t,self.objFun(instances),t*X.shape[0],fn))
            j=random.randint(0, len_f-1)
            k=random.randint(0, len_g-1)
            g_jk=0
            b_k=0       
            for i,x in enumerate(instances):
                if x.feature_vector[j] and x.gammas[k]:  
                   db=-x.c*x.label*x.gammas[k]/(1+np.exp(x.label*self.Z[i]))
                   g_jk+=x.feature_vector[j]*db
                   b_k+=db
            if div_by_cc:
                g_jk/=cc
                b_k/=cc
            change_w=self.W[j][k]
            change_b=self.bias[k]
            
            if (self.W[j][k]-(g_jk/beta))>l_div_b:
                self.W[j][k]=self.W[j][k]-(g_jk/beta)-l_div_b
                
            elif (self.W[j][k]-(g_jk/beta))<-l_div_b:
                self.W[j][k]=self.W[j][k]-(g_jk/beta)+l_div_b                
            else:
                self.W[j][k]=0
                
            if not self.no_bias:
                
                self.bias[k]=self.bias[k]-(b_k/beta)

            
            change_w=self.W[j][k]-change_w
            change_b=self.bias[k]-change_b
            
            if change_w or change_b:
                for i,x in enumerate(instances):  
                    if x.feature_vector[j] and x.gammas[k]:      
                        self.Z[i]+=change_w*(x.feature_vector[j]*x.gammas[k])+change_b*(x.gammas[k])
        self.iters += T
                    
    def localWb(self,instances,**kwargs):
        """
        Compute the local weight and bias for a given gamma
        """
        if type(instances)!=type([]): #if it is a single instance
            instances = [instances]
        
         
        if self.encoder is not None and not len(instances[0].gammas):       
            gammas = np.vstack([self.encoder.encode(np.atleast_2d(i.feature_vector),**kwargs) for i in instances]).T
            return np.dot(np.vstack((self.W,self.bias)),gammas)
        else:
            gammas=np.vstack([(np.atleast_2d(i.gammas)) for i in instances])
            return np.dot(np.vstack((self.W,self.bias)),gammas.T)
        
        
    def predict_instance(self, instance, **kwargs):

       score=None
       X=instance.feature_vector
       G=np.atleast_2d(instance.gammas).T
       score=(self.W.dot(G)).T.dot(X)+G.T.dot(self.bias)
       

       return score

    def __str__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)
        
    def __logHistory__(self,val):
        self.history.append(val)
        
    def objFun(self,instances):
        E=np.sum([i.c*np.log(1+np.exp(-self.score(i)*i.label)) for i in instances])
        R=np.linalg.norm(self.W.ravel(),ord=1)
        T = self.Lambda*R/self.W.shape[1] + E
        return (T,R,E)
