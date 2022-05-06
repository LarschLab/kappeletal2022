# -*- coding: utf-8 -*-
"""
Created on Mon Dec 05 10:27:39 2016

@author: jlarsch
"""
import numpy as np
import math
#import cv2
#from matplotlib import pyplot as plt





def tailfit_function(frame, head, tail_length):
    
    arc_points=40 #number of points on full circle
    threshold=240 #don't trust tail pixels above threshold. Extend previous direction instead.
    tail_points = [head]
    tail_points_cv = [(head[0], head[1])]
    num_points = 5
    width = tail_length
    x = head[0]
    y = head[1]
    img_filt = np.zeros(frame.shape)
    #img_filt = cv2.boxFilter(frame, -1, (2,2), img_filt)
    img_filt=frame
    
    #pre compute full circle to look up pixel values
    #this needs a reasonable guess for 'arc_points' or will result in
    #duplicate points that mess up location of minimum/maximum
    #ideally, this should be calculated using rasterized mid point cirlce  
    
    lin = np.linspace(0,2*np.pi,arc_points)
    ls_all=(width/num_points*np.sin(lin)).astype(int);
    lc_all=(width/num_points*np.cos(lin)).astype(int);
    
    #use lookup_offset to rotate lookup circle depending on where tail was found
    lookup_offset=0
    d=1 #for first point after head, search full circle. After that, half circle
    for j in range(num_points):
        try:
            #print(lookup_offset)
            ls=np.roll(ls_all,-lookup_offset)[0:int(arc_points/d)]
            lc=np.roll(lc_all,-lookup_offset)[0:int(arc_points/d)]
            d=2 #beginning from second point, search half circle
            # Find the x and y values of the arc
            xs = x+ls
            ys = y+lc

            #use min/max, depending on inverted image
            # ident = np.where(img_filt[ys,xs]==max(img_filt[ys,xs]))[0][0]
            ident = np.where(img_filt[ys,xs]==min(img_filt[ys,xs]))[0][0]
            
            #special treatment for low contrast end of tail
            if img_filt[ys[ident],xs[ident]]>threshold:
                ident=int(arc_points/4) #use center of arc to extend tail in previous direction
            
            #update lookup offset
            lookup_offset=lookup_offset-(int(arc_points/4)-ident)

            # print ident
            x = xs[ident]
            y = ys[ident]

            # Add point to list
            tail_points.append([x,y])
            tail_points_cv.append((x,y))
            
        except IndexError:
            # tail_points.append([np.nan,np.nan])
            # tail_points_cv.append((np.nan,np.nan))
            tail_points.append(tail_points[-1])
            tail_points_cv.append(tail_points_cv[-1])
    tailangle = float(math.atan2(np.nanmean(np.asarray(tail_points)[-3:-1,1])-np.asarray(tail_points)[0,1],np.nanmean(np.asarray(tail_points)[-3:-1,0])-np.asarray(tail_points)[0,0])*180.0/3.1415)
    return np.asarray(tail_points), tailangle*-1, tail_points_cv
    

#
#avi_path= 'D:\\data\\b\\2017\\20170914_embeddedRepeat\\out_id0_150fps_20170915102458_unc.avi'
##img = cv2.imread(image_path)
#
#capture = cv2.VideoCapture(avi_path)
#tf=[]
#nFrames=130000#150*60*30
#print 'analyzing frames',nFrames
#for i in range(nFrames):
#    if np.mod(i,10000)==0:
#        print 'reached frame',i
#    capture.set(cv2.CAP_PROP_POS_FRAMES,i)
#    img = capture.read()[1]
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    #blur = cv2.boxFilter(img,(4,4))
#    img=cv2.boxFilter(img, 0, (5,5), img, (-1,-1), True, cv2.BORDER_DEFAULT)
#    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img)  
#    tf.append(tailfit_function(img,min_loc,45.0)[1])

