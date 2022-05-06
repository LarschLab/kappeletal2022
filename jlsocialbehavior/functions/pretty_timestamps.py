# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:35:06 2016

class to convert pointgrey timestamps into seconds
from goncalo lopes, kampff lab

@author: jlarsch
"""

import numpy as np
import pandas as pd
import tkFileDialog
import os
import matplotlib.pyplot as plt

p = tkFileDialog.askopenfilename(initialdir=os.path.normpath('\\\\O1-103\\e\\00_bonsai_ffmpeg_out'))    
df=pd.read_csv(p,index_col=None,header=None)

t=df.values[:,0]
#tc=converttime(t)
#tcu=uncycle(tc)
plt.plot((t-t[0])/10000000.0)
iv=np.diff(t/10000000.0)
plt.figure()
plt.hist(iv,bins=np.arange(0.0055,0.07,0.0001))
plt.yscale('log')
print (t[-1]-t[0])/10000000.0

def converttime(time):
    #offset = time & 0xFFF
    cycle1 = (time >> 12) & 0x1FFF
    cycle2 = (time >> 25) & 0x7F
    seconds = cycle2 + cycle1 / 8000.
    return seconds
    
def uncycle(time):
    cycles = np.insert(np.diff(time) < 0, 0, False)
    cycleindex = np.cumsum(cycles)
    return time + cycleindex * 128
