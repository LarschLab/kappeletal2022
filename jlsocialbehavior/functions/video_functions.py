# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 15:34:22 2016

@author: jlarsch
"""

import numpy as np
import os
import functions.gui_circle as gc
import functions.ImageProcessor as ImageProcessor
import models.geometry as geometry
import cv2
import pandas as pd
import glob
import sys

from functions.getMedVideo import getMedVideo


def get_pixel_scaling(aviPath,forceCorrectPixelScaling=0,forceInput=0,bg_file=''):
    #pixel scaling file will typically reside in parent directory where the raw video file lives
    #forceCorrectPixelScaling=0 - force user iput if no previous data exists
    #forceInput=0 - force user input even if data exists - overwrite
    head, tail = os.path.split(aviPath)
    head=os.path.normpath(head)
    try:
        bg_file=glob.glob(head+'\\dishImage*.jpg')[0]
    except:
        #print 'no background image found for pixel scaling, regenerating...'
        bg_file= getMedVideo(aviPath)[1]
    
    parentDir = os.path.dirname(head)
    scaleFile = os.path.join(parentDir,'bgMed_scale.csv')
    
    if np.equal(~os.path.isfile(scaleFile),-2) or forceCorrectPixelScaling:
        #aviPath = tkFileDialog.askopenfilename(initialdir=parentDir,title='select video to generate median for scale information')
        bg_file= getMedVideo(aviPath, bg_file=bg_file)[1]
#        print bg_file, 'run circleGUI'
        scaleData=gc.get_circle_rois(bg_file,'_scale',forceInput)[0]        
      
    elif forceInput or (np.equal(~os.path.isfile(scaleFile),-1) and  forceCorrectPixelScaling):
        scaleData=np.array(np.loadtxt(scaleFile, skiprows=1,dtype=float))
    else:
        print('no PixelScaling found, using 8 pxPmm')
        return 8

    pxPmm=2*scaleData['circle radius']/scaleData['arena size']
    return pxPmm.values[0]
        
def getAnimalLength(aviPath,frames,coordinates,boxSize=200,threshold=20,invert=False):
    cap = cv2.VideoCapture(aviPath)
    #vp=getVideoProperties(aviPath)
    #videoDims = tuple([int(vp['width']) , int(vp['height'])])
#    print videoDims
    eAll=np.zeros((frames.shape[0],coordinates.shape[1],6))
    for i in range(frames.shape[0]): #use FramesToAvg images to calculate median
        string= str(i)+' out of '+ str(frames.shape[0])+' frames.'
        sys.stdout.write('\r'+string) 
        
        f=frames[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES,f)
        image=cap.read()
        try:
            gray = cv2.cvtColor(image[1], cv2.COLOR_BGR2GRAY)
        except:
            gray=image[1]
        for j in range(coordinates.shape[1]):
            g=gray.copy()
            if np.isnan(coordinates[i,j,0]):
                eAll[i,j,:]=np.nan
                #print i,'nan at frame ',f,j
            else:
                currCenter=geometry.Vector(*coordinates[i,j,:].astype('int'))
                crop=ImageProcessor.crop_zero_pad(g,currCenter,boxSize)
       
                if invert:
                    crop=255-crop
                img_binary = ImageProcessor.to_binary(crop.copy(), threshold,invertMe=False)            
                im_tmp2,contours, hierarchy = cv2.findContours(img_binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_center=geometry.Vector(crop.shape[0]/2,crop.shape[1]/2)
    
                cnt = ImageProcessor.get_contour_containing_point(contours,img_center)
                try:            
                    (x,y),(MA,ma),ori=cv2.minAreaRect(cnt[0])
                    eAll[i,j,0:5]=[x,y,MA,ma,ori]
                    eAll[i,j,0:2]=eAll[i,j,0:2]+currCenter
                    mask=crop.copy()*0
                    cv2.drawContours(mask, cnt[0], -1, (255),-1)
                    eAll[i,j,5]=cv2.mean(crop,mask=mask)[0]
                except:
                    #plt.figure()
                    #plt.imshow(crop)
                    #print cnt
                    #print 'position',currCenter
                    #print i,'problem at frame ',f,j
                    eAll[i,j,:]=np.nan

            
    return eAll      


def getAnimalSize(experiment,needFrames=2000,numFrames=40000,boxSize=200,e2=[]):
    avi_path=experiment.expInfo.aviPath
    head, tail = os.path.split(avi_path)
    sizeFile = os.path.join(head,'animalSize.txt')    
    

    
    if ~np.equal(~os.path.isfile(sizeFile),-2):
        print('determining animalSize from data')
        haveFrames=0
        frames=np.zeros(needFrames).astype('int')
        dist=np.zeros(needFrames)
        
        triedFr=[]
        triedD=[]
        while haveFrames<needFrames:
            tryFrame=np.random.randint(1000,numFrames,1)
            minDist=np.max(np.abs(np.diff(experiment.rawTra[tryFrame,:,:],axis=1)))
            if minDist>boxSize:
                frames[haveFrames]=int(tryFrame)
                dist[haveFrames]=minDist
                haveFrames += 1
            else:
                triedFr.append(tryFrame)
                triedD.append(minDist)
        
        
        tra=experiment.rawTra[frames,:,:]
        if e2!=[]:
            tra[:,0,:]=experiment.rawTra[frames,1,:]

            tra[:,1,:]=e2.rawTra[frames,1,:]
            tra[:,0,0]=tra[:,0,0]+512
            #tra[:,:,1]=512-tra[:,:,1]
            print('using shifted secondAnimal trajectory')
        
        #if (int(experiment.expInfo.videoDims[0])/float(tra.max()))>2:
            
        
        tmp=getAnimalLength(avi_path,frames,tra)
        brightness=np.mean(tmp[:,:,5],axis=0).astype('int')
        MA=np.max(tmp[:,:,2:4],axis=2)
        bins=np.linspace(0,100,101)
        anSize=[np.argmax(np.histogram(MA[:,0],bins=bins)[0]),np.argmax(np.histogram(MA[:,1],bins=bins)[0])]
    
        df=pd.DataFrame({'anID':[1,2],'anSize':anSize,'brightness':brightness},index=None)
        df.to_csv(sizeFile,sep='\t',index=False)
        anID=np.array([1,2])

        ret=np.vstack([anID,anSize]).T     
    else:
#        print 'loading saved animalSize'
        tmp = pd.read_csv(sizeFile, dtype=int, delim_whitespace=True, skipinitialspace=True)
        
        ret = np.array(tmp[[0, 1]].values)  # np.array(np.loadtxt(sizeFile, skiprows=1,dtype=int))
        
    return ret
    
