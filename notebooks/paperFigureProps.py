# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:38:07 2017

@author: jlarsch
"""




import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
#import pylab
inToCm=2.54
figSq=(5,5)
figLs=(5,3)
figPt=(3,5)



def paper():

    
    print('using plot settings for paper plots')
    print('standard figure sizes \n figSq: ',figSq,'\n figLs: ',figLs,'\n figPt: ',figPt)
    inToCm=2.54
    
    sns.set()
    sns.set_style("ticks")
    sns.set_context("paper")
    
    mpl.rcParams['lines.linewidth'] = 1
    mpl.rcParams['figure.autolayout'] = False
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['axes.titlesize'] = 10
    mpl.rcParams['figure.titlesize'] = 10
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['lines.linewidth'] = 1.0
    mpl.rcParams['lines.markersize'] = 10
    mpl.rcParams['legend.fontsize'] = 8
    mpl.rcParams['figure.dpi'] = 150
    mpl.rcParams['savefig.dpi'] = 150
    mpl.rcParams['figure.figsize'] = figSq
    mpl.rcParams['svg.fonttype'] = 'none'