# -*- coding: utf-8 -*-
"""
Created on Tue May 22 12:45:43 2018

@author: jlarsch
"""

# bonsai stores 'corrected animal position' corresponding to projector coordinates in the csv output.
# correction is computet using intercept theorem
# correction is subtracted from (x,y) coordinates


# to find animals in the camera frame, need to 'deCorrect' the coordinates by adding back the correction factor

#need same parameters that were used during the experiment. This is currently hard coded!

def deCorrectFish(x,y,xOff,yOff,xMax,yMax,camHeight):#camHeight default used to be 79
  
  xR=x+xOff
  yR=y+yOff

  xR=xR+((xR-xMax/2.)/(camHeight))
  yR=yR+((yR-yMax/2.)/(camHeight))

  return xR-xOff,yR-yOff

def CorrectFish(x,y,xOff,yOff,xMax,yMax,camHeight): #camHeight default used to be 79
  
  xR=x+xOff
  yR=y+yOff

  xR=xR-((xR-xMax/2.)/(camHeight))
  yR=yR-((yR-yMax/2.)/(camHeight))

  return xR-xOff,yR-yOff