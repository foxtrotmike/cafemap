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
    
    fname='data/prostate_preprocessed.txt'
    X,Y, genes=readData(fname, 'tumor')
    X = (X.T/np.linalg.norm(X,axis=1)).T
    instances=createInstances(X, Y)
    llc = compute_gammas(instances, K=10, k=10)
    c = cafeMap(Lambda = 1e-3, T = 10e2, no_bias = True, encoder = llc)    
    result, folds = c.kFoldCV(instances,parallel=1)
    scores,labels,classifiers = zip(*result)
    classifier = classifiers[0]
    Wb = classifiers[0].localWb(instances)#[:-1]
    for c in classifiers[1:]:
        Wb+=c.localWb(instances)#[:-1]
    
        
    print [np.mean(np.sum(np.abs(x.localWb(instances)[:-1])>0,axis = 0)) for x in classifiers]
    perFoldAuc, perFoldAcc= perFoldAUC(scores, labels)
    print "The AVG AUC for 10 folds=", np.mean(perFoldAuc)
    print "The AVG Accuracy (zero threshold) for 10 folds=", np.mean(perFoldAcc)


    
    from sklearn.cluster import KMeans
    Wb0 = Wb*1
    Wb = 100*Wb
    idx = np.sum(np.abs(Wb)>1e-6,axis = 1)>0
    idx = np.argsort(np.sum(np.abs(Wb),axis = 1))[-40:]
    Wbr = Wb[idx,:]
    Wbr = Wbr[:,Y==-1]
    model = KMeans(init='k-means++',n_clusters=5)
    model.fit(Wbr[:-1,:].T)
    cc = model.cluster_centers_
    cci = model.labels_[np.argmax(np.bincount(model.labels_))]
    dd = dict(zip(np.argsort([np.linalg.norm(cc[cci]-c) for c in cc]),range(model.n_clusters)))
    lbln = np.argsort([dd[i] for i in model.labels_])
    Wbrn = Wbr[:,lbln]
    
    Wbr = Wb[idx,:]
    Wbr = Wbr[:,Y==1]
    model = KMeans(init='k-means++',n_clusters=5)
    model.fit(Wbr[:-1,:].T)
    cc = model.cluster_centers_
    cci = model.labels_[np.argmax(np.bincount(model.labels_))]
    dd = dict(zip(np.argsort([np.linalg.norm(cc[cci]-c) for c in cc]),range(model.n_clusters)))
    lblp = np.argsort([dd[i] for i in model.labels_])
    Wbrp = Wbr[:,lblp]
    
    
    vmin,vmax = np.min(Wb[:-1,:]),np.max(Wb[:-1,:])
    cmap = plt.cm.jet; #plt.cm.plasma
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.matshow(Wbrn,vmin=vmin,vmax = vmax, interpolation = 'none');
    ax1.set_title("Negative Examples")
    vv = ax2.matshow(Wbrp,vmin=vmin,vmax = vmax, interpolation = 'none');
    ax2.set_title("Positive Examples")
    ax1.set_ylabel("Selected Features")
    fig.subplots_adjust(right=0.91, top = 0.87, left = 0.04, hspace = 0.20, wspace = 0.01)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
    fig.colorbar(vv, cax=cbar_ax)
    
    idx=idx[:-1]
    vmin,vmax = np.min(X),np.max(X)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    vv = ax1.matshow(X.T[idx,:][:,Y==-1][:,lbln],vmin=vmin,vmax = vmax);
    ax1.set_title("Negative Examples")
    ax2.matshow(X.T[idx,:][:,Y==1][:,lblp],vmin=vmin,vmax = vmax);
    ax2.set_title("Positive Examples")
    ax1.set_ylabel("Selected Features")
    fig.subplots_adjust(right=0.91, top = 0.87, left = 0.04, hspace = 0.20, wspace = 0.01)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
    fig.colorbar(vv, cax=cbar_ax)

#well technically it isnt biclustering
#its a plot of the top ranking features
#the examples have been clustered based on their top scoring local features
#this reveals that there are actually different clusters of examples
#The weight value of an example is high then this means there is a similar example
#of the opposite class close to it. If an example is significaly different from examples 
#of the other class, then its weight values are small