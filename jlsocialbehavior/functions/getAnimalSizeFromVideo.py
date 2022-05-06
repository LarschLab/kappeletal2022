# -*- coding: utf-8 -*-
"""
Created on Tue May 22 12:40:52 2018

@author: jlarsch
"""
import numpy as np
import os

import functions.getVideoProperties
import functions.video_functions as vf
import functions.CameraInterceptCorrection as cic

def getAnimalSizeFromVideo(currAvi,rawData,camHeight, sizePercentile=40,numPairs=15, roiPath=[]):
    videoInfo = functions.getVideoProperties.getVideoProperties(currAvi)
    xMax = int(videoInfo['width']) #relevant for openGL scaling
    yMax = int(videoInfo['height'])

    numFrames=1000#2000
    maxFrames=100000
    boxSize=200
    head, tail = os.path.split(currAvi)
    print('processing animals size:', currAvi)
    
    head, tail = os.path.split(roiPath)
    #print tail
    if tail[:3]=='ROI':
        rois=np.loadtxt(roiPath)
        correct=True
        
        #for virtual pairing, can use random frames, no need to avoid collisions
        traAll=np.zeros((numFrames,numPairs,2))
        frames=np.random.randint(1000,20000,numFrames)
        print('using random frames, decorrected position to determine size. correction: ',xMax,yMax,camHeight)
        for i in range(numPairs):
            currCols=[i*3,i*3+1]
            rawTra=rawData[rawData.columns[currCols]].values
            tra=rawTra.copy()
            xx=rawTra[:,0]
            yy=rawTra[:,1]
            xoff=rois[i,0]
            yoff=rois[i,1]
            if camHeight > 0:
                xx,yy=cic.deCorrectFish(xx,yy,xoff,yoff,xMax,yMax,camHeight)
            tra[:,0]=xx+xoff
            tra[:,1]=yy+yoff
            traAll[:,i,:]=tra[frames,:].copy()           
        
    else:
        print('using non-collisions and raw position to determine size.')
        currCols=[0,1,3,4]
        rawTra=rawData[rawData.columns[currCols]].values.reshape(-1,2,2)
        rois=np.loadtxt(roiPath,skiprows=1,delimiter=',')
        correct=False
    
        haveFrames=0
        frames=np.zeros(numFrames).astype('int')
        dist=np.zeros(numFrames)
        
        triedFr=[]
        triedD=[]
        print('determining 2000 frames to read animal size...')
        while haveFrames<numFrames:
            tryFrame=np.random.randint(1000,maxFrames,1)
            minDist=np.max(np.abs(np.diff(rawTra[tryFrame,:,:],axis=1)))
            if minDist>boxSize:
                frames[haveFrames]=int(tryFrame)
                dist[haveFrames]=minDist
                haveFrames += 1
            else:
                triedFr.append(tryFrame)
                triedD.append(minDist)
        
        print('done. tried',len(triedFr),'frames')
        traAll=np.zeros((numFrames,numPairs,2))
        for i in range(numPairs):
            currCols=[i*3,i*3+1]
            rawTra=rawData[rawData.columns[currCols]].values
            tra=rawTra.copy()
            traAll[:,i,:]=tra[frames,:].copy()           



    invert=(not correct)
    print('inverting video',invert)
    tmp=vf.getAnimalLength(currAvi,frames,traAll,threshold=5,invert=invert)

    anSize=[]
    MA=[]
    for i in range(numPairs):
        MA.append(np.max(tmp[:,i,2:4],axis=1))
        anSize.append(np.nanpercentile(MA[i],sizePercentile))
    
    fnSizeAll=head+'\\anSizeAll.csv'
    #print 'writing size to',fnSizeAll
    with open(fnSizeAll,'wb') as f:
        np.savetxt(f,np.array(MA),fmt='%.5f')
        
    return np.array(anSize)#/self.pxPmm

