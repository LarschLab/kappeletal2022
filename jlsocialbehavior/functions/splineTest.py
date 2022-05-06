# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 08:52:58 2016

@author: jlarsch
"""

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import networkx as nx
import numpy as np
from scipy import interpolate
import models.geometry as geometry
import functions.ImageProcessor as ImageProcessor
import matplotlib.pyplot as plt

#img_binary = ImageProcessor.to_binary(img_crop.copy(), 240)

def return_skeleton(img_binary):
#skeletonize image and return an ordered list that represents the shortest path through the skeleton points
#this is going to be trouble when there is significant branching
    img_bin_crop=img_binary
    skel=ImageProcessor.to_skeleton(img_bin_crop)
    try:
        skel_smooth,skel_list, ep=get_skeleton_list(skel,30,150)
    except:
        skel_smooth,skel_list, ep = None,None,None
        
    if skel_smooth is not None:
        skel_angles=get_line_angles(skel_smooth)
    else:
        skel_angles=np.zeros(29).tolist()
    return skel_angles, skel_smooth,skel, ep
    
def get_skeleton_list(skel,number_points,x_offset=0):
    #get skeleton pixel coordinates and fit a smooth spline to it
    #coordinates need to be sorted along the skeleton
    #sorting via shortest path through nearest neighbor graph

    #x_offset allows cropping the skeleton to omit parts of it for fitting
    #currently shifting to the contour centroid.
    #this circumvents the problem of skeleton branching in the head.
    #This only makes sense if the animal is already rotated to face left.

    skel_list=np.where(skel[:,x_offset:]>0)
    points=np.c_[skel_list[0],skel_list[1]+x_offset]
    ep=findEndpoints(skel[:,x_offset:])
    
    if len(ep)==2: #there are more endpoints when two animals collide
        #correct endpoints for offset to match original skeleton
        for i in range(np.shape(ep)[0]):
            ep[i,1]=ep[i,1]+x_offset
        
        #create graph G of 2 nearest neighbors for each point.    
        G=NN_graph(points)   
         
    #------alternative strategy to fill the graph-------- not working yet
    #    clf = NearestNeighbors(2).fit(points,2)
    #    G = clf.kneighbors_graph()
    #    T = nx.from_scipy_sparse_matrix(G)
        
        #find shortest path through the graph from one end-point to the other
        #then fit a b-spline to the sorted list and return a list of interpolated points on that fit.

        sp=nx.shortest_path(G,source=(ep[0,0],ep[0,1]),target=(ep[1,0],ep[1,1]))
        spa=np.array(sp)
        x=spa[:,0]
        y=spa[:,1]
        s=5
        k=1
        nest=-1
        tckp,u=interpolate.splprep([x,y],s=s,k=k,nest=nest)
        xnew,ynew=interpolate.splev(np.linspace(0,1,number_points),tckp)
        return np.c_[xnew,ynew], spa, ep

    else: #animal collision: more than 2 endpoints
        return None, None, None


def NN_graph(points):
    G = nx.Graph()  # A graph to hold the nearest neighbours
    xyl=tuple(map(tuple,points))
    tree = KDTree(points, leaf_size=2, metric='euclidean')  # Create a distance tree
    #for p in xyl[1:-1]:  # Skip first and last items in list
    for p in xyl:
        dist, ind = tree.query(np.reshape(p,(1,-1)), k=3)
        # ind Indexes represent nodes on a graph
        # Two nearest points are at indexes 1 and 2. 
        # Use these to form edges on graph
        # p is the current point in the list
        G.add_node(p)
        n1 = xyl[ind[0][1]]  # The next nearest point
        n2 = xyl[ind[0][2]]  # The following nearest point
        G.add_edge(p, n1)
        G.add_edge(p, n2)
    return G
    

def findEndpoints(img_skel):
    #finding endpoints in a binary skeleton image by looking at pixels with only one neighbor.
    endPoints=[]
    sb_list=np.where(img_skel>0)
    
    #looking inside the 9 pixel squares centered on each point skeleton point
    for i in range(np.shape(sb_list)[1]):
        xmin=np.max([0,sb_list[0][i]-1])
        xmax=np.min([sb_list[0][i]+2,img_skel.shape[0]])
        ymin=np.max([0,sb_list[1][i]-1])
        ymax=np.min([sb_list[1][i]+2,img_skel.shape[1]])
        n=img_skel[xmin:xmax,ymin:ymax]
        nn=np.sum(n>0)
        if nn<3:
            endPoints.append([sb_list[0][i],sb_list[1][i]])
    #sort endPoints by x-position
    endPoints=np.array(endPoints)
    endPoints=endPoints[endPoints[:,1].argsort()]
    return endPoints
        
def get_line_angles(points):
    #angle between adjacent points in a list
    points_base=points[:-1]
    points_fwd=points[1:]
    line_angles=[]
    for j in range(points_base.shape[0]):
        v1=geometry.Vector(*points_base[j])
        v2=geometry.Vector(*points_fwd[j])      
        line_angles.append(geometry.Vector.get_angle(v1,v2))
    return line_angles

