# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 13:30:45 2016

@author: jlarsch
"""

import numpy as np
import functions.peakdet as peakdet
import models.geometry as geometry
import matplotlib.pyplot as plt
import scipy.stats as sta

class AnimalBoutSeriesCollection:
    def __init__(self):
        #parameters for bout analysis
        #(orientation begins to jitter a couple of frames before bout peak)
        self.t=2 #average window come this close to bout start
        self.n=5 #how many frames to average for bout analysis
    
    def linkAnimal(self,animal):
        self.animal=animal
        animal.add_BoutSeriesCollection(self)
        
        #detect bouts as peaks in tail curvature.
        #omit first and last 10 frames to avoid 'out of range' problems with downstream processing of data around peaks.
        self.bout_start=peakdet.detect_peaks(self.animal.ts.tailCurvature_mean()[10:-10],1,8)+10
    
        
    def rv_before_bout(self):
        #median range vector over n frames up to frame t before bout begins
        #get range vector for frames
        rv=self.animal.ts.rangeVector()
        x_list=[rv[b-(self.t+self.n):b-self.t] for b in self.bout_start]
        #median
        x=np.nanmedian(x_list,axis=1)
        return x
        
    def heading_before_bout(self):
        #median heading angle over n frames up to frame t before bout begins
        #get heading angle for frames
        ori=self.animal.ts.fish_orientation_elipse()
        x_list=[ori[b-(self.t+self.n):b-self.t] for b in self.bout_start]
        #median
        x=np.nanmedian(x_list,axis=1)
        return x
   
    def heading_after_bout(self):
        #median heading angle over n frames up to frame t before bout begins
        #get heading angle for frames
        ori=self.animal.ts.fish_orientation_elipse()
        x_list=[ori[b+self.t:b+self.t+self.n] for b in self.bout_start]
        #median
        x=np.nanmedian(x_list,axis=1)
        return x
        
    def error_before_bout(self):
        #this is the angle between median range vector before and heading before the bout
        x=geometry.smallest_angle_difference_degrees(self.heading_before_bout(),self.rv_before_bout()) 
        return x
    
    def error_after_bout(self):
        #this is the angle between median range vector BEFORE and heading after the bout
        #this represents the error with respect to the location of the neighbor BEFORE the bout
        #rv before bout is the information the animal had to initiate the turn
        x=geometry.smallest_angle_difference_degrees(self.heading_after_bout(),self.rv_before_bout()) 
        return x
        
    def d_error_inst(self):
        #absolute change in error before vs. after bout
        x=np.diff(np.abs([self.error_before_bout(),self.error_after_bout()]).T,axis=1)
        return np.squeeze(x)
        
    def d_heading(self):
        #x=np.squeeze(np.diff([self.heading_before_bout(),self.heading_after_bout()],axis=0).T)
        x=geometry.smallest_angle_difference_degrees(self.heading_before_bout(),self.heading_after_bout())
        return x
        
    def d_IAD_lastBout(self):
        dist=self.animal.pair.IAD()
        x=[0]
        x.extend([dist[b]-dist[b-1] for b in self.bout_start[1:]])
        return np.squeeze(np.array(x))