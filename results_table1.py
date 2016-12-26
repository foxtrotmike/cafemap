# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 16:49:51 2015

@author: Amina Asif
"""

from cafeMap import *
import numpy as np
def readData(fname, pos_label):
    with open(fname,'r') as ifile:

        feat_vecs=[]
        labels=[]
        genes=[]
        
    
        for ln in ifile:
            ln = ln.strip()
           
            ln=ln.split(' ')
            if ln[0][0]=='y':
                for l in ln[1:]:
                    if l==pos_label:
                        labels+=[1.0]
                    else:
                        labels+=[-1.0]
            else:    
                vector=[]
                for f in ln[1:]:
                    vector+=[np.float(f)]
                feat_vecs+=[vector]
                genes+=[ln[0]]            
            
        return np.array(feat_vecs).T, np.array(labels), genes

if __name__ == '__main__':
    
    
    
   #========================lymphoma==================================#
    fname='data/dlbcl_preprocessed.txt'
    features,labels, genes=readData(fname, '1')
    instances=createInstances(features, labels)
    compute_gammas(instances, K=10, k=10, gamma=0.1)
    c=cafeMap(T=100000, Lambda=0.0001, beta=0.1)
    result, folds= c.kFoldCV(instances,  parallel=4)
    scores,labels,classifiers = zip(*result)
    perFoldAuc, perFoldAcc, perFoldBestAcc, perFoldThresh= perFoldAUC(scores, labels)
    print "The AVG AUC for 10 folds(Lymphoma)=", np.mean(perFoldAuc)
    print "The AVG Accuracy (zero threshold) for 10 folds(Lymphoma)=", np.mean(perFoldAcc)
    print "The AVG Accuracy for 10 folds(Lymphoma best threshold)=", np.mean(perFoldBestAcc) # (best threshold)
  
#===============================breast cancer=====================#
#    
    fname='data/breast_preprocessed.txt'
    features,labels, genes=readData(fname, 'luminal')    
    instances=createInstances(features, labels)
    compute_gammas(instances, K=10, k=10, gamma=10.0)
    c=cafeMap( T=100000, Lambda=0.01, beta=0.1)
    result, folds= c.kFoldCV(instances, parallel=4)
    scores,labels,classifiers = zip(*result)
    perFoldAuc, perFoldAcc, perFoldBestAcc, perFoldThresh= perFoldAUC(scores, labels)
    print "The AVG AUC for 10 folds(Breast cancer)=", np.mean(perFoldAuc)
    print "The AVG Accuracy (zero threshold) for 10 folds(Breast Cancer)=", np.mean(perFoldAcc)
    print "The AVG Accuracy for 10 folds(Breast Cancer best threshold)=", np.mean(perFoldBestAcc) # (best threshold)
  

#==========================Prostate Cancer=====================#
    fname='data/prostate_preprocessed.txt'
    X,Y, genes=readData(fname, 'tumor')
    instances=createInstances(X, Y)    
    llc = compute_gammas(instances, K=10, k=10, gamma=10.0)
    c = cafeMap(Lambda=0.00001, beta=0.1, T = 1e5)    
    result, folds = c.kFoldCV(instances,parallel=4)
    scores,labels,classifiers = zip(*result)
    perFoldAuc, perFoldAcc, perFoldBestAcc, perFoldThresh= perFoldAUC(scores, labels)
    print "The AVG AUC for 10 folds(Prostate Cancer)=", np.mean(perFoldAuc)
    print "The AVG Accuracy (zero threshold) for 10 folds(Prostate Cancer)=", np.mean(perFoldAcc)
    print"The AVG Accuracy for 10 folds(Prostate Cancer best threshold)=", np.mean(perFoldBestAcc)
 