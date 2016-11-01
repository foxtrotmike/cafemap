# -*- coding: utf-8 -*-
"""
Created on Mon Apr 04 15:54:50 2016

@author: afsar
"""
from random import random as rand
from math import pi, cos, sin
import numpy as np
import matplotlib.pyplot as plt
def getRT():
    r = rand()
    theta = (2*rand()-1)*pi
    return (r,theta)
def getCircle(n = 50):
    P = []
    N = []
    for i in range(n):
        (r,theta) = getRT()
        x = [r*cos(theta), r*sin(theta)]
        P.append(x)
        (r,theta) = getRT()
        r+=1.2
        x = [r*cos(theta), r*sin(theta)]
        
        N.append(x)
    
    P = np.array(P)
    N = np.array(N)
    return P,N
if __name__=='__main__':
    P,N = getCircle()
    plt.plot(P[:,0],P[:,1],'+')
    plt.plot(N[:,0],N[:,1],'o')