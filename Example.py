# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 21:53:20 2016

@author: Amina Asif
This file demonstrates the use of cafemap implementation
"""



from cafeMap import*


def readData(fname, pos_label):
    feat_vecs=[]
    labels=[]
    genes=[]
    with open(fname,'r') as ifile:  
           
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
    
    fname='data\\prostate_preprocessed.txt'
    features,labels, genes=readData(fname, 'tumor') # 'tumor' will be considered as +1 label
    
    
    instances=createInstances(features, labels)
    #instances= data points of type Instance as needed by cafemap
    
    c=cafeMap(T=10000, Lambda=0.01, beta=0.1, K=10, k=10, gamma=0.1)
    # T= number of iterations
    #Lambda= regularization parameter
    #beta= beta parameter in coordinate descent algorithm
    #K= number of Anchor points in llc
    #k= number of non zero coefficients
    #gamma= hyper parameter >0 for llc to enforce sparsity and locality 
    
    result, folds= kFoldCV(c, instances)
    # c= trained cafemap classifier
    
    #result, folds= kFoldCV(c, instances, parallel=4)
    # parallel= number of Cpu cores to be used
    # parallel implementation requires "joblib"
    
    scores,labels,classifiers = zip(*result)
    perFoldAuc, perFoldAcc= perFoldAUC(scores, labels)
    print "The AVG AUC for 10 folds=", np.mean(perFoldAuc)
    print "The AVG Accuracy for 10 folds=", np.mean(perFoldAcc)
