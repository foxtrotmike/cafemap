# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 01:51:29 2015

@author: amina
"Cross Validation Module"
contains:
Class definition for 'fold'
create_folds
trainTest
cv
test
"""
import random
from itertools import chain
import numpy as np
from copy import deepcopy
from sklearn import metrics
#from sklearn import metrics
from .roc import *
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]
def chunkify(lst,n):
    return [ lst[i::n] for i in xrange(n) ]    
    
def perFoldAUC(dec_scores, labels):
    AUCs=[]
    ACCs=[]
    maxACCs=[]
    thresholds=[]
    
    for i in range(len(dec_scores)):
        fpr, tpr, auc= roc(dec_scores[i], labels[i])
        f, t, a=metrics.roc_curve(labels[i], dec_scores[i])
        AN=sum(x<0 for x in labels[i])
        AP=sum(x>0 for x in labels[i])
        TN=(1.0-f)*AN
        TP=t*AP
        Acc2=(TP+TN)/len(labels[i])
        thresh_ind=np.argmax(Acc2)
        thresh=a[thresh_ind]
        thresholds+=[thresh]
        maxACCs+=[max(Acc2)]
        
        
        
        
        
        AUCs+=[auc]
        Acc=0.0
       
        for j in range(len(dec_scores[i])):
            if (dec_scores[i][j]>0 and labels[i][j]>0) or(dec_scores[i][j]<0 and labels[i][j]<0):
                Acc+=1.0
        Acc=Acc/len(dec_scores[i])
        ACCs+=[Acc]
        
         
#        import pdb; pdb.set_trace()
        
    return AUCs, ACCs, maxACCs, thresholds

def AUC(dec_scores, labels):
    dec_scores=list(np.array(dec_scores).flatten())
    labels=list(np.array(labels).flatten())
    
    fpr, tpr, auc= roc(dec_scores, labels)
    return auc
        

class fold:    
    """
    Contains a list of indices for training and testing instances
    fold.train_instances: list (of indices) of training instances
    fold.test_instances: list (of indices) of testing instances
    """
    def __init__(self):
        self.train_instances=[]
        self.test_instances=[]
        
def separate_instances_multi(instances, classes):
    sep_instances=[]
    for c in classes:
        l_b=[]
        for ind in range (len(instances)):
            if instances[ind].label==c:
                l_b+=[ind]
        sep_instances+=[l_b]
    return sep_instances
      
def separate_instances(instances): 
    """
    Seperates the positive and negative instances
    takes a list of instances as input and returns list of indices for positive and negative instances
    pos, neg=separate_instances(instances)
    """
#    random.shuffle(instances)
    pos_instances=[]
    neg_instances=[]
    for ind in range (len(instances)):
        if instances[ind].label==1.0:
            pos_instances+=[ind]
        else:
            neg_instances+=[ind]
    return pos_instances, neg_instances
 
           
def create_folds(instances, no_of_folds, **kwargs):
    """
    Creates folds from the given data.
    Takes a list of instances and the desired number of folds as input.
    Returns a list of fold objects
    """
    pos, neg=separate_instances(instances)
    shuffle = kwargs.get('shuffle',True)
    if shuffle:
        random.shuffle(pos)
        random.shuffle(neg)
        
#    n=len(neg)/no_of_folds
#    p=len(pos)/no_of_folds
    pos_chunks = chunkify(neg,no_of_folds)#list(chunks(pos,p))
    neg_chunks = chunkify(pos,no_of_folds)#list(chunks(neg,n))
    folds = []
    for i in range(no_of_folds):
        f = fold()
        f.train_instances = pos_chunks[:i]+pos_chunks[i+1:]+neg_chunks[:i]+neg_chunks[i+1:]
        f.train_instances = list(chain(*f.train_instances)) #flatten list of lists
        f.test_instances = pos_chunks[i]+neg_chunks[i]        
        folds.append(f)
    return folds 

def trainTest(classifier,train_instances,test_instances, **kwargs):
    """
    Trains the classifier over train_instances and returns the decision scores of test_instances
    Takes a classifier object, list of training instances and list of test instances as input
    Returns the list of decision scores    
    """    
    classifier.train(train_instances, **kwargs)
    return classifier.test(test_instances, **kwargs)
    
def CV(classifier_temp,instances,folds=10, **kwargs):
    def CVone(f, **kwargs):        
        train_instances=[];  
        for index in range(len(f.train_instances)):          
            train_instances+=[instances[int(f.train_instances[index])] ]
        test_instances=[]
        for index in range(len(f.test_instances)):
            test_instances+=[instances[int(f.test_instances[index])] ]
        classifier=classifier_temp.__class__(deepcopy(classifier_temp))
        dec_scores = classifier.trainTest(train_instances,test_instances, **kwargs)
        labels=[]        
        labels+=[b.label for b in test_instances]
        return dec_scores, labels, classifier
        
    if type(folds) != type([]): #when there is only one fold, no need to pass a list
        return CVone(folds, **kwargs) 
        
    if 'parallel' in kwargs and kwargs['parallel']>1:        
        numproc = kwargs['parallel'] 
        if numproc: # if user wants parallelism   
            from joblib import Parallel, delayed #import only when it's needed              
            print("Using",numproc,"Processors")
            result = Parallel(n_jobs=numproc, verbose = True)\
                (delayed(CV)(classifier_temp,instances,f,**kwargs) for f in folds) 
    else:        
        result = [CVone(f, **kwargs) for f in folds]        
        
    return result
    
    
################################################################################################   
   
def kFoldCV(classifier_temp,instances,folds = 10, **kwargs):
    if type(folds)==type(0):
        folds=create_folds(instances,folds,**kwargs)    
    return CV(classifier_temp,instances,folds, **kwargs),folds
    
  
def LOOCV(classifier_temp,instances, **kwargs):
    folds=[]
    for i in range(len(instances)):
        f=fold()
        f.test_instances+=[i]
        if i != 0:
            f.train_instances+=range(0,i)
        
        if i !=(len(instances)-1):
            f.train_instances+=range(i+1, len(instances))
            
        folds.append(f)
    return CV(classifier_temp,instances,folds, **kwargs)
    
###################################################################################################            


def test(classifier,data, **kwargs):
    """
    Test a classifier over a list of instances
    Takes classifier object and list of instances as data
    Returns a list of decision scores
    """    
    scores = [classifier.score(i, **kwargs) for i in data]
    return scores
    




