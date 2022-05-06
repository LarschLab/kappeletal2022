# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 20:52:49 2015

@author: johannes
"""
import numpy as np
import os

def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho
    
def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y
    
def runningMean(x, N):
    return np.convolve(x, np.ones((N,))/N,mode='valid')[(N-1):]

def distance(x,y):
    return np.sqrt(x**2 + y**2)

def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

def equalizePath(x,y,precision=2):
    

    M = len(x)*100
    t = np.linspace(0, len(x), M)
    xi = np.interp(t, np.arange(len(x)), x)
    yi = np.interp(t, np.arange(len(y)), y)
    
    
    i, idx = 0, [0]
    while i < len(xi)-1:
        total_dist = 0
        for j in range(i+1, len(xi)):
            total_dist = np.sqrt((xi[j]-xi[i])**2 + (yi[j]-yi[i])**2)
            if total_dist > precision:
                idx.append(j)
                break
        i = j+1
    
    xn = xi[idx]
    yn = yi[idx]
    
    # Interpolate values for x and y.
    t = np.arange(len(xn))
    t2 = np.linspace(0, len(xn), len(x))
    # One-dimensional linear interpolation.
    xnn = np.interp(t2, t, xn)
    ynn = np.interp(t2, t, yn)
    return xnn,ynn
    
def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """


    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is not one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    

    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    if x.ndim > 2:
        raise ValueError("smooth only accepts 1 or 2 dimension arrays.")

    if x.ndim == 1:
        s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        yAll=np.convolve(w/w.sum(),s,mode='valid')[int(round(window_len/2-1, 0)):-int(round(window_len/2, 0))]
        
    if x.ndim == 2:
        y=[]
        for c in range(x.shape[1]):
            s=np.r_[x[window_len-1:0:-1,c],x[:,c],x[-2:-window_len-1:-1,c]]
            y.append(np.convolve(w/w.sum(),s,mode='valid')[int(round(window_len/2-1, 0)):-int(round(window_len/2, 0))])
        yAll=np.array(y).T
    
    return yAll