import os
import pickle
import subprocess

import numpy as np


def getVideoProperties(aviPath):

    #check if videoData has been read already
    head, tail = os.path.split(aviPath)
    head=os.path.normpath(head)
    videoPropsFn=os.path.join(head,'videoProps.pickle')

    if np.equal(~os.path.isfile(videoPropsFn),-1):
        #read video metadata via ffprobe and parse output
        #can't use openCV because it reports tbr instead of fps (frames per second)
        cmnd = ['c:/ffmpeg/bin/ffprobe', '-show_format', '-show_streams', '-pretty', '-loglevel', 'quiet', aviPath]
        p = subprocess.Popen(cmnd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        out = str(out)[3:-10]
        decoder_configuration = {}
        for line in out.split('\\r\\n'):
            if '=' in line:
                key, value = str(line).split('=')
                decoder_configuration[key] = value

        #frame rate needs special treatment. calculate from parsed str argument
        nominator, denominator = decoder_configuration['avg_frame_rate'].split('/')
        decoder_configuration['fps'] = int(float(nominator) / float(denominator))

        #save video data for re-use
        with open(videoPropsFn, 'wb') as f:
            pickle.dump([decoder_configuration], f)

    else:


        with open(videoPropsFn, 'rb') as f:
            decoder_configuration=pickle.load(f)[0]
        #print('re-using VideoProps')

    return decoder_configuration