# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:29:28 2016

@author: jlarsch
"""
#This function generates uniformly distributed dots on a circle

#Inputs---
# rad: radius of circle
# num: number of dots

#outputs---
#dotList: numpy array of x,y coordinates of dots
#dotND: numpy array of all pairwise neighbor distances

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import scipy.spatial


def randomDotsOnCircle(rad=50, num=1000):

    t = np.random.uniform(0.0, 2.0*np.pi, num)
    r = rad * np.sqrt(np.random.uniform(0.0, 1.0, num))
    x = r * np.cos(t)
    y = r * np.sin(t)
    
    #plt.plot(x, y, "ro", ms=1)
    #plt.axis([-rad, rad, -rad, rad])
    #plt.show()
    
    a=[x,y]
    dotList=np.array(a).T
    tree=scipy.spatial.cKDTree(dotList,leafsize=10)
    
    TheResult=[]
    for item in dotList:
    
       TheResult.append(tree.query(item, k=num, distance_upper_bound=5+(2*rad))[0])
    
    c=np.array(TheResult)
    
    dotND=c.flatten()

    return dotList,dotND